"""동일 teacher setup에서 바람 프로그램을 순차 실행하고 모든 시도를 기록한다.

Newton은 실행 함수 안에서만 import한다. Suite 검사와 program compilation은 NumPy core다.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Sequence
import uuid

from .physics_registry import ArrayIdentity, TeacherPhysicsRegistry, canonical_json_bytes, content_hash
from .initial_state import TeacherInitialDisplacement
from .sample_meshes import SampleClothMesh
from .trajectory import MAX_CHUNK_FRAMES, arrays_hash, require, writer_contract
from .trajectory_io import (
    TeacherTrajectoryArtifact, _atomic_json, _file_hash, _json_load, _path, _read_arrays,
    inspect_teacher_run, _write_arrays,
)
from .wind_programs import COMPILER_POLICY, WindProgram

if TYPE_CHECKING:
    from .newton_cloth import NewtonClothConfig


SUITE_SCHEMA = "wind3dgs.teacher_wind_suite.v1"
PLAN_SCHEMA = "wind3dgs.teacher_wind_suite_plan.v1"
DISPLACED_SUITE_SCHEMA = "wind3dgs.teacher_wind_suite.v2"
DISPLACED_PLAN_SCHEMA = "wind3dgs.teacher_wind_suite_plan.v2"
CASE_STATES = {"not_started", "running", "completed", "failed", "interrupted", "io_failed", "aborted"}
SUITE_STATES = {"running", "completed", "completed_with_failures", "interrupted", "io_failed", "aborted"}
PHYSICAL_FAILURES = {
    "traction_guard", "nonfinite", "pin_drift", "pin_velocity", "extent", "degenerate_face",
    "preroll_not_converged", "preroll_nonfinite", "preroll_pin_drift",
}
EXECUTION_POLICY = "sequential_fresh_initial_state_continue_physics_failure_stop_infrastructure_no_retry_v1"


def _counts(cases: list[dict]) -> dict:
    counts = {state: sum(c["status"] == state for c in cases) for state in sorted(CASE_STATES)}
    return {"declared": len(cases), "attempted": len(cases) - counts["not_started"], **counts}


def _hash(manifest: dict) -> str:
    return content_hash({k: v for k, v in manifest.items() if k != "manifest_sha256"})


def _checkpoint(root: Path, manifest: dict) -> None:
    manifest["counts"] = _counts(manifest["cases"])
    manifest["all_runs_passed"] = (manifest["status"] == "completed"
                                    and manifest["counts"]["completed"] == manifest["counts"]["declared"])
    manifest["manifest_sha256"] = _hash(manifest)
    _atomic_json(root / "suite_manifest.json", manifest)


def _case_path(root: Path, index: int) -> Path:
    runs = root / "runs"
    path = runs / f"case_{index:06d}"
    require(not runs.is_symlink() and not path.is_symlink(), "suite_path", "run directory symlink를 허용하지 않습니다")
    return path


def _plan_cases(programs: Sequence[WindProgram], fps: int) -> list[dict]:
    cases = []
    for index, program in enumerate(programs):
        compiled = program.compile(fps)
        cases.append({"index": index, "program_id": program.program_id, **compiled.description(),
                      "run_dir": f"runs/case_{index:06d}", "status": "not_started", "failure": None,
                      "run_id": None, "run_manifest_sha256": None, "run_content_sha256": None})
    return cases


def _load_plan(root: Path, manifest: dict) -> dict:
    plan = _json_load(_path(root, "plan.json").read_bytes())
    require(content_hash(plan) == manifest["plan_sha256"], "suite_plan_hash", "실행 계획 hash 불일치")
    displaced = manifest["schema_version"] == DISPLACED_SUITE_SCHEMA
    fields = {"schema_version", "compiler_policy", "execution_policy", "config", "registry", "programs", "seed", "chunk_frames"}
    if displaced:
        fields.add("initial_displacement")
    require(set(plan) == fields, "suite_plan_fields", "실행 계획 필드 불일치")
    require(plan["schema_version"] == (DISPLACED_PLAN_SCHEMA if displaced else PLAN_SCHEMA)
            and plan["compiler_policy"] == COMPILER_POLICY
            and plan["execution_policy"] == EXECUTION_POLICY, "suite_plan_version", "미지원 실행 계약")
    require(type(plan["config"]) is dict and type(plan["programs"]) is list and bool(plan["programs"]),
            "suite_plan_type", "setup과 프로그램 목록이 필요합니다")
    require(type(plan["seed"]) is int and plan["seed"] >= 0 and type(plan["chunk_frames"]) is int
            and 1 <= plan["chunk_frames"] <= MAX_CHUNK_FRAMES, "suite_plan_type", "seed/chunk 범위")
    registry = TeacherPhysicsRegistry.from_dict(plan["registry"])
    require(displaced == registry.schema_version.endswith(".v2"), "suite_plan_version", "registry/suite version 불일치")
    if displaced:
        entry = plan["initial_displacement"]
        require(type(entry) is dict and set(entry) == {"sha256", "content_sha256", "bytes"},
                "initial_displacement_fields", "초기 변위 inventory 필드 불일치")
        file = _path(root, "initial_displacement.npz")
        require(type(entry["bytes"]) is int and file.stat().st_size == entry["bytes"], "initial_displacement_size", "변위 파일 크기 불일치")
        _verify_displacement_arrays(_read_arrays(root, "initial_displacement.npz", entry), registry)
    return plan


def _verify_displacement_arrays(arrays: dict, registry: TeacherPhysicsRegistry) -> None:
    require(set(arrays) == {"displacement_m"}, "initial_displacement_fields", "요청 변위 배열 불일치")
    require(ArrayIdentity.from_array(arrays["displacement_m"], unit="m") == registry.initial_state.requested_displacement,
            "initial_displacement_identity", "suite/child 요청 변위 identity 불일치")


def inspect_teacher_wind_suite(path: str | Path) -> dict:
    """중단·진행 중인 suite도 계획/분모를 확인한다. 자식 run 검증은 open()이 소유한다."""
    root = Path(path)
    manifest = _json_load(_path(root, "suite_manifest.json").read_bytes())
    require(set(manifest) == {"schema_version", "suite_id", "created_at", "status", "plan_sha256",
                              "cases", "counts", "all_runs_passed", "failure", "manifest_sha256"},
            "suite_fields", "suite manifest 필드 불일치")
    require(manifest["schema_version"] in (SUITE_SCHEMA, DISPLACED_SUITE_SCHEMA)
            and type(manifest["status"]) is str and manifest["status"] in SUITE_STATES,
            "suite_version_status", "미지원 suite schema/status")
    require(manifest["manifest_sha256"] == _hash(manifest), "suite_manifest_hash", "suite manifest 손상")
    plan = _load_plan(root, manifest)
    registry = TeacherPhysicsRegistry.from_dict(plan["registry"])
    if manifest["status"] in {"completed", "completed_with_failures"}:
        require(1.0 / plan["config"]["fps"] == registry.solver.frame_dt_s,
                "suite_time_grid", "config와 registry의 frame dt 불일치")
    programs = [WindProgram.from_dict(p) for p in plan["programs"]]
    require(len({p.program_id for p in programs}) == len(programs), "duplicate_program_id", "중복 program ID")
    expected = _plan_cases(programs, plan["config"]["fps"])
    cases = manifest["cases"]
    require(type(cases) is list and len(cases) == len(expected), "suite_denominator", "선언한 프로그램이 누락되었습니다")
    mutable = {"status", "failure", "run_id", "run_manifest_sha256", "run_content_sha256"}
    for case, original in zip(cases, expected):
        require(type(case) is dict and set(case) == set(original), "suite_case_fields", "case 필드 불일치")
        require(all(case[k] == v for k, v in original.items() if k not in mutable), "suite_case_identity", "계획과 case 불일치")
        require(type(case["status"]) is str and case["status"] in CASE_STATES, "suite_case_status", "미지원 case 상태")
        _case_path(root, case["index"])
        for key in ("run_manifest_sha256", "run_content_sha256"):
            value = case[key]
            require(value is None or (type(value) is str and len(value) == 64
                                      and all(c in "0123456789abcdef" for c in value)), "suite_run_hash", key)
        if case["status"] in {"not_started", "running"}:
            require(all(case[k] is None for k in mutable - {"status"}), "suite_case_state", "미완료 case에 결과가 있습니다")
        elif case["status"] == "completed":
            require(case["failure"] is None and all(case[k] is not None for k in
                    ("run_id", "run_manifest_sha256", "run_content_sha256")), "suite_case_state", "완료 case identity 누락")
        else:
            require(type(case["failure"]) is dict, "suite_case_state", "실패 사유 누락")
            if case["status"] == "failed":
                require(case["failure"]["code"] in PHYSICAL_FAILURES and case["run_manifest_sha256"] is not None,
                        "suite_failure_class", "검증 가능한 물리 실패만 다음 run으로 진행합니다")
    suffix_only = False
    for case in cases:
        if suffix_only:
            require(case["status"] == "not_started", "suite_execution_order", "중단/미실행 case 이후 실행 기록이 있습니다")
        suffix_only = suffix_only or case["status"] in {"not_started", "running", "interrupted", "io_failed", "aborted"}
    require(manifest["counts"] == _counts(cases), "suite_counts", "실패/미실행 분모 불일치")
    completed = manifest["status"] in {"completed", "completed_with_failures"}
    if completed:
        require(manifest["failure"] is None and all(c["status"] in {"completed", "failed"} for c in cases),
                "suite_completion", "아직 처리되지 않은 case가 있습니다")
        require((manifest["status"] == "completed") == all(c["status"] == "completed" for c in cases),
                "suite_completion", "suite 성공 상태 불일치")
    elif manifest["status"] == "running":
        require(manifest["failure"] is None, "suite_failure_state", "진행 중 상태와 종료 사유 불일치")
    else:
        require(type(manifest["failure"]) is dict and set(manifest["failure"]) == {"code", "stage", "case_index"},
                "suite_failure_state", "중단 사유 누락")
        index = manifest["failure"]["case_index"]
        require(index is None or (type(index) is int and 0 <= index < len(cases)
                                  and cases[index]["status"] == manifest["status"]),
                "suite_failure_state", "중단된 case index 불일치")
    require(type(manifest["all_runs_passed"]) is bool
            and manifest["all_runs_passed"] == (manifest["status"] == "completed"), "suite_success", "전체 성공 표시 불일치")
    return manifest


def _verify_child(root: Path, case: dict, plan: dict) -> None:
    path = _case_path(root, case["index"])
    run = inspect_teacher_run(path)
    require(run["manifest_sha256"] == case["run_manifest_sha256"] and run["run_id"] == case["run_id"]
            and run["content_sha256"] == case["run_content_sha256"], "suite_child_identity", "자식 run identity 변경")
    registry = TeacherPhysicsRegistry.from_dict(plan["registry"])
    require(run["registry_sha256"] == registry.registry_hash and run["seed"] == plan["seed"]
            and run["config_sha256"] == content_hash(plan["config"])
            and run["expected_intervals"] == case["interval_count"], "suite_child_setup", "자식 setup 불일치")
    require(run["writer_contract"] == writer_contract(registry)
            and run["schema_version"] == writer_contract(registry)["schema_version"], "suite_child_setup", "자식 schema 불일치")
    if registry.schema_version.endswith(".v2"):
        require("initial_displacement.npz" in run["outputs"], "initial_state_input", "자식 요청 변위 파일 누락")
        _verify_displacement_arrays(_read_arrays(path, "initial_displacement.npz", run["outputs"]["initial_displacement.npz"]), registry)
    if case["status"] == "completed":
        artifact = TeacherTrajectoryArtifact.open(path)
        require(arrays_hash(artifact.wind) == case["samples_sha256"], "suite_child_wind", "프로그램과 실제 바람 불일치")
    else:
        require(run["status"] == "failed" and run["failure"] == case["failure"], "suite_child_failure", "실패 사유 불일치")
        # 실패 prefix/진단도 inventory에서 삭제되거나 바뀌지 않았는지 확인한다.
        for name, entry in run["outputs"].items():
            file = _path(path, name)
            require(file.is_file() and file.stat().st_size == entry["bytes"] and _file_hash(file) == entry["sha256"],
                    "suite_failed_artifact_hash", name)
        require("wind.npz" in run["outputs"], "suite_child_wind", "실패 run의 wind 입력 누락")
        wind = _read_arrays(path, "wind.npz", run["outputs"]["wind.npz"])
        require(arrays_hash(wind) == case["samples_sha256"], "suite_child_wind", "실패 run의 wind 입력 불일치")


def _verify_children(root: Path, cases: list[dict], plan: dict) -> None:
    run_ids = set()
    for case in cases:
        _verify_child(root, case, plan)
        require(case["run_id"] not in run_ids, "duplicate_run_id", "서로 다른 case가 같은 run을 참조합니다")
        run_ids.add(case["run_id"])


@dataclass(frozen=True)
class TeacherWindSuiteArtifact:
    path: Path
    manifest: dict

    @property
    def all_runs_passed(self) -> bool:
        return self.manifest["all_runs_passed"]

    @classmethod
    def open(cls, path: str | Path) -> TeacherWindSuiteArtifact:
        root = Path(path)
        manifest = inspect_teacher_wind_suite(root)
        require(manifest["status"] in {"completed", "completed_with_failures"},
                "incomplete_suite", "실행이 끝난 suite만 열 수 있습니다")
        plan = _load_plan(root, manifest)
        _verify_children(root, manifest["cases"], plan)
        return cls(root, manifest)


def _capture_child(root: Path, case: dict) -> dict | None:
    """실패 처리 중 손상/부재가 원래 예외를 가리지 않도록 metadata만 복사한다."""
    try:
        run = inspect_teacher_run(_case_path(root, case["index"]))
    except (OSError, ValueError, KeyError):
        return None
    case.update(run_id=run["run_id"], run_manifest_sha256=run["manifest_sha256"],
                run_content_sha256=run["content_sha256"])
    return run


def run_teacher_wind_suite(mesh: SampleClothMesh, config: NewtonClothConfig,
                           registry: TeacherPhysicsRegistry, programs: Sequence[WindProgram],
                           output_dir: str | Path, *, chunk_frames: int = 16, seed: int = 0,
                           initial_displacement: TeacherInitialDisplacement | None = None) -> TeacherWindSuiteArtifact:
    """한 object/setup에서 순서대로 새 run을 생성한다. 재시도·resume·split 배정은 하지 않는다.

    물리 실패는 분모에 남기고 다음 case를 실행한다. 입력/실행기 오류, I/O 오류와 중단은
    남은 case를 not_started로 보존하고 예외를 다시 발생시킨다.
    """
    from .newton_physics_registry import build_teacher_physics_registry
    from .newton_trajectory import TeacherRunFailed, record_teacher_run

    programs = tuple(programs)
    require(bool(programs) and all(type(p) is WindProgram for p in programs), "programs", "WindProgram 목록이 필요합니다")
    require(len({p.program_id for p in programs}) == len(programs), "duplicate_program_id", "중복 program ID")
    require(type(seed) is int and seed >= 0 and type(chunk_frames) is int and 1 <= chunk_frames <= MAX_CHUNK_FRAMES,
            "suite_input", "seed/chunk 범위 불일치")
    cases = _plan_cases(programs, config.fps)
    displaced = registry.schema_version.endswith(".v2")
    require(displaced == (initial_displacement is not None), "initial_state_input", "registry와 suite 초기 변위를 함께 지정해야 합니다")
    require(initial_displacement is None or type(initial_displacement) is TeacherInitialDisplacement,
            "initial_state_input", "TeacherInitialDisplacement가 필요합니다")
    plan = {"schema_version": DISPLACED_PLAN_SCHEMA if displaced else PLAN_SCHEMA,
            "compiler_policy": COMPILER_POLICY, "execution_policy": EXECUTION_POLICY,
            "config": asdict(config), "registry": registry.to_dict(), "programs": [p.to_dict() for p in programs],
            "seed": seed, "chunk_frames": chunk_frames}
    # JSON 의미로 정규화하고 mutable mesh 배열은 suite 시작에 한 번 복사한다.
    plan = _json_load(canonical_json_bytes(plan))
    mesh = replace(mesh, vertices=mesh.vertices.copy(), faces=mesh.faces.copy(), uv=mesh.uv.copy(),
                   pinned=mesh.pinned.copy(), pin_groups=mesh.pin_groups.copy(), metadata=deepcopy(mesh.metadata))
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    if initial_displacement is not None:
        plan["initial_displacement"] = _write_arrays(root / "initial_displacement.npz", {
            "displacement_m": initial_displacement.displacement_numpy(),
        })
    manifest = {"schema_version": DISPLACED_SUITE_SCHEMA if displaced else SUITE_SCHEMA, "suite_id": uuid.uuid4().hex,
                "created_at": datetime.now(timezone.utc).isoformat(), "status": "running",
                "plan_sha256": content_hash(plan), "cases": cases, "counts": _counts(cases),
                "all_runs_passed": False, "failure": None}
    _atomic_json(root / "plan.json", plan)
    _checkpoint(root, manifest)
    active = None
    stage = "setup_validation"
    try:
        expected = build_teacher_physics_registry(
            mesh, config, source_object_id=registry.source.source_object_id,
            object_group_id=registry.source.object_group_id, split_manifest_ref=registry.source.split_manifest_ref,
            material_preset_ref=registry.material_preset_ref,
            initial_displacement=initial_displacement,
        )
        require(expected.registry_hash == registry.registry_hash, "registry_mismatch", "suite setup과 registry 불일치")
        for case, program in zip(cases, programs):
            stage = "compile"
            compiled = program.compile(config.fps)
            description = compiled.description()
            require(description == {key: case[key] for key in description},
                    "compiled_wind_changed", "계획 이후 wind sample이 바뀌었습니다")
            active = case
            case["status"] = "running"
            _checkpoint(root, manifest)
            stage = "run"
            try:
                artifact = record_teacher_run(mesh=mesh, config=config, registry=registry,
                                              wind_samples=compiled.samples, output_dir=_case_path(root, case["index"]),
                                              chunk_frames=chunk_frames, seed=seed, initial_displacement=initial_displacement)
            except TeacherRunFailed as error:
                run = _capture_child(root, case)
                if error.code not in PHYSICAL_FAILURES or run is None or run["status"] != "failed":
                    raise
                require(run["failure"]["code"] == error.code, "failure_code_mismatch", "예외와 run 실패 사유 불일치")
                case.update(status="failed", failure=run["failure"])
                stage = "verify_failed_run"
                _verify_child(root, case, plan)
            else:
                require(artifact.path.resolve() == _case_path(root, case["index"]).resolve(), "suite_child_path", "자식 출력 경로 불일치")
                case.update(status="completed", run_id=artifact.manifest["run_id"],
                            run_manifest_sha256=artifact.manifest["manifest_sha256"],
                            run_content_sha256=artifact.manifest["content_sha256"])
                require(artifact.registry.registry_hash == registry.registry_hash
                        and artifact.manifest["config_sha256"] == content_hash(plan["config"])
                        and arrays_hash(artifact.wind) == case["samples_sha256"],
                        "suite_child_setup", "자식 run과 suite 입력 불일치")
            _checkpoint(root, manifest)
            active = None
        stage = "final_validation"
        # 완료 표시 전에 저장된 모든 child를 대조한다. 디스크 손상을 성공으로 발행하지 않는다.
        inspect_teacher_wind_suite(root)
        _verify_children(root, cases, plan)
        manifest["status"] = "completed_with_failures" if any(c["status"] == "failed" for c in cases) else "completed"
        _checkpoint(root, manifest)
        return TeacherWindSuiteArtifact(root, manifest)
    except BaseException as error:
        if isinstance(error, OSError):
            status, code = "io_failed", "io_failure"
        elif isinstance(error, (KeyboardInterrupt, SystemExit)):
            status, code = "interrupted", "interrupted"
        else:
            status, code = "aborted", getattr(error, "code", "execution_failed")
        manifest["status"] = status
        manifest["failure"] = {"code": code, "stage": stage, "case_index": None if active is None else active["index"]}
        if active is not None:
            run = _capture_child(root, active)
            active["status"] = status
            active["failure"] = (run["failure"] if run is not None and run["failure"] is not None
                                 else {"code": code, "stage": stage})
        try:
            _checkpoint(root, manifest)
        except OSError:
            # 기록 불가 디스크에서는 마지막 running checkpoint와 원래 예외를 보존한다.
            pass
        raise
