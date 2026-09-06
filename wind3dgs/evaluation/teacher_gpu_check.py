"""사용자가 직접 실행하는 CUDA Teacher smoke 검사와 로그 supervisor.

직접 파일 실행의 supervisor 경로는 표준 라이브러리만 사용한다. Optional import/CUDA
초기화와 native crash는 자식 프로세스에서 격리하여 실패 로그를 보존한다.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
import traceback


CODE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = CODE_ROOT.parent
CASE_PLAN = (
    ("wind_mesh2_sub4", 2, 4, False), ("wind_mesh4_sub4", 4, 4, False),
    ("wind_mesh8_sub4", 8, 4, False), ("wind_mesh8_sub8", 8, 8, False),
    ("wind_mesh8_sub16", 8, 16, False), ("decay_mesh4_sub4", 4, 4, True),
    ("decay_mesh8_sub4", 8, 4, True),
)
STAGES = tuple(name for name, *_ in CASE_PLAN) + (
    "compare_spatial", "compare_temporal", "compare_decay_spatial", "replay_wind", "replay_decay",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".pending")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _redact(text: str, output: Path) -> str:
    prefixes = {str(WORKSPACE): "<workspace>", str(Path.home()): "<home>",
                str(Path(sys.prefix).resolve()): "<python-env>", str(output.resolve()): "<run>"}
    for prefix in sorted(prefixes, key=len, reverse=True):
        if len(prefix) > 1:
            text = text.replace(prefix, prefixes[prefix])
    return text


def _snapshot() -> dict:
    packages = {}
    for name in ("numpy", "newton", "warp-lang"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    sources = [*CODE_ROOT.glob("wind3dgs/teacher/*.py"), *CODE_ROOT.glob("wind3dgs/evaluation/*.py"),
               CODE_ROOT / "scripts/check_teacher_gpu.sh", CODE_ROOT / "pyproject.toml"]
    return {"python": platform.python_version(), "os": platform.system(), "machine": platform.machine(),
            "packages": packages,
            "sources_sha256": {str(p.relative_to(CODE_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(sources)}}


def _nvidia_smi() -> dict:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return {"return_code": result.returncode, "gpu_csv": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"return_code": None, "status": type(error).__name__}


def _select_cuda(device_name: str, output: Path) -> str:
    import warp as wp

    wp.init()
    devices = wp.get_cuda_devices()
    info = {"requested_device": device_name, "cuda_driver_version": wp.get_cuda_driver_version(),
            "cuda_devices": [{"alias": str(d), "name": str(d.name), "arch": d.arch} for d in devices],
            "cuda_visible_devices_is_set": "CUDA_VISIBLE_DEVICES" in os.environ}
    _json_write(output / "cuda.json", info)
    if not devices or device_name not in {str(d) for d in devices}:
        raise RuntimeError("요청한 CUDA device를 사용할 수 없습니다. CPU fallback은 수행하지 않습니다.")
    device = wp.get_device(device_name)
    if not device.is_cuda:
        raise RuntimeError("CUDA device가 필요합니다.")
    # 실제 device allocation/host transfer를 수행하여 열거만 성공한 상태와 구분한다.
    value = wp.zeros(1, dtype=wp.float32, device=device)
    wp.synchronize_device(device)
    if value.numpy().tolist() != [0.0]:
        raise RuntimeError("CUDA 메모리 왕복 검사에 실패했습니다.")
    print(f"CUDA 준비 완료: {device} / {device.name}", flush=True)
    return str(device)


def _execute_checks(output: Path, device: str, checks: dict) -> None:
    """작은 고정 fixture의 전체 경로. CPU 입력은 이 함수의 자동 검사에서만 사용한다."""
    import numpy as np
    from wind3dgs.evaluation import (
        TeacherConvergenceReport, TeacherConvergenceSpec, TeacherRefinementRun, compare_teacher_refinements,
    )
    from wind3dgs.teacher import (
        ArtifactReference, ProbeMappingPolicy, TeacherProbeSet, WindSample, build_teacher_probe_map,
        extract_teacher_probe_trajectory, make_cantilever_initial_displacement, make_sample_mesh,
    )
    from wind3dgs.teacher.newton_cloth import NewtonClothConfig
    from wind3dgs.teacher.newton_physics_registry import build_teacher_physics_registry
    from wind3dgs.teacher.newton_trajectory import record_teacher_run, replay_teacher_run
    from wind3dgs.teacher.physics_registry import SourceObjectScope, content_hash
    from wind3dgs.teacher.trajectory import require

    def stage(name, action):
        checks["active_stage"] = name
        checks["stages"][name] = {"status": "running", "started_at": _now()}
        _json_write(output / "checks.json", checks)
        started = time.monotonic()
        print(f"검사 시작: {name}", flush=True)
        result = action()
        checks["stages"][name].update(status="passed", elapsed_s=time.monotonic() - started, result=result)
        _json_write(output / "checks.json", checks)
        print(f"검사 통과: {name}", flush=True)

    scope = SourceObjectScope("gpu-smoke-flag", "gpu-smoke-flag",
                              ArtifactReference("development-smoke-only", content_hash({"fixture": "teacher_gpu_smoke_v1"})))
    points = np.array([(x, 0., z) for z in np.linspace(-.5, .5, 5) for x in np.linspace(0, 1, 5)])
    weights = np.array([.5, 1., 1., 1., .5]) / 4
    ids = [f"p{i:04d}" for i in range(25)]
    ids[14] = "tip_center"
    probes = TeacherProbeSet(source=scope, probe_ids=ids, rest_positions_m=points,
                             area_weights_m2=np.outer(weights, weights).ravel(), reference_mass_kg=.1)
    policy = ProbeMappingPolicy(coverage_tolerance_m=1e-7, barycentric_tolerance=1e-12,
                                partition_tolerance=2e-15, affine_reproduction_tolerance=1e-12,
                                quadrature_relative_tolerance=1e-6)
    runs = {}
    def record(name, resolution, substeps, displaced):
        mesh = make_sample_mesh("rectangular_flag", width_m=1., height_m=1., resolution=(resolution, resolution))
        config = NewtonClothConfig(run_mode="teacher", device=device, reference_mass_kg=.1,
                                   fps=60, substeps=substeps, iterations=10,
                                   initial_state_policy="displaced_gravity_off" if displaced else "gravity_off",
                                   air_drag_enabled=not displaced)
        initial = make_cantilever_initial_displacement(mesh, amplitude_m=.01) if displaced else None
        registry = build_teacher_physics_registry(mesh, config, source_object_id=scope.source_object_id,
                                                  object_group_id=scope.object_group_id, split_manifest_ref=scope.split_manifest_ref,
                                                  initial_displacement=initial)
        source = record_teacher_run(mesh=mesh, config=config, registry=registry, initial_displacement=initial,
                                    wind_samples=[WindSample(.5)] * 6 + [WindSample(0., False)] * 6,
                                    output_dir=output / "runs" / name / "raw", chunk_frames=5)
        require(source.manifest["device"] == device, "gpu_check_device", "실제 실행 device가 요청과 다릅니다")
        mapping = build_teacher_probe_map(mesh, probes, policy=policy)
        sampled = extract_teacher_probe_trajectory(source.path, mapping, output / "runs" / name / "probe")
        require(sampled.source_verified, "gpu_check_source", "원본/probe 대조가 필요합니다")
        peak_speed, pin_drift, guard_count, absolute_external_work = 0., 0., 0, 0.
        for chunk in source.iter_chunks():
            peak_speed = max(peak_speed, float(np.linalg.norm(chunk["velocities_m_s"], axis=2).max()))
            pin_drift = max(pin_drift, float(np.abs(chunk["positions_m"][:, mesh.pinned] - mesh.vertices[mesh.pinned]).max()))
            guard_count += int(chunk["guard_count"].sum())
            absolute_external_work += float(np.abs(chunk["external_work_j"]).sum())
        require(peak_speed > 0 and pin_drift == 0 and guard_count == 0, "gpu_check_health", "운동/고정점/guard 검사를 통과하지 못했습니다")
        if displaced:
            require(absolute_external_work == 0, "gpu_check_decay", "aero-off 자유감쇠의 외력 work는 0이어야 합니다")
        runs[name] = TeacherRefinementRun(name, source.path, sampled.path)
        return {"device": device, "raw": f"runs/{name}/raw", "probe": f"runs/{name}/probe",
                "raw_content_sha256": source.manifest["content_sha256"], "registry_sha256": registry.registry_hash,
                "source_verified": True, "interval_count": 12, "peak_speed_m_s": peak_speed,
                "max_pin_drift_m": pin_drift, "guard_count": guard_count, "absolute_external_work_j": absolute_external_work}

    for name, resolution, substeps, displaced in CASE_PLAN:
        stage(name, lambda: record(name, resolution, substeps, displaced))

    def compare(name, keys, axis, displaced=False):
        spec = TeacherConvergenceSpec(axis=axis, tip_probe_ids=("tip_center",), reference_length_m=1., reference_time_s=1.,
                                      frequency_bands_hz=((0., 15.), (15., 30.)),
                                      initial_condition="cantilever_quadratic" if displaced else "gravity_off_rest",
                                      initial_amplitude_m=.01 if displaced else None)
        selected = [runs[k] for k in keys]
        report = compare_teacher_refinements(selected, spec)
        target = output / "comparisons" / name
        report.save(target)
        checked = TeacherConvergenceReport.open(target, runs=selected)
        require(checked.source_verified, "gpu_check_comparison", "비교 원본 재계산을 확인해야 합니다")
        return {"path": f"comparisons/{name}", "report_sha256": checked.report_hash, "source_verified": True,
                "convergence_status": checked.to_dict()["convergence_status"]}

    stage("compare_spatial", lambda: compare("spatial", STAGES[:3], "spatial"))
    stage("compare_temporal", lambda: compare("temporal", STAGES[2:5], "temporal"))
    stage("compare_decay_spatial", lambda: compare("decay_spatial", STAGES[5:7], "spatial", True))
    def replay(name):
        report = replay_teacher_run(runs[name].source_run_dir, device=device)
        require(report.passed and report.device == device, "gpu_check_replay", "같은 device에서 수치 재생을 통과하지 못했습니다")
        return asdict(report)
    stage("replay_wind", lambda: replay(STAGES[4]))
    stage("replay_decay", lambda: replay(STAGES[6]))


def _worker(output: Path, device: str) -> int:
    sys.path.insert(0, str(CODE_ROOT))
    checks = {"schema_version": "wind3dgs.teacher_gpu_check.v1", "status": "running", "requested_device": device,
              "active_stage": "cuda_preflight", "failure": None,
              "stages": {name: {"status": "not_started"} for name in STAGES}, "convergence_status": "not_assessed"}
    _json_write(output / "checks.json", checks)
    try:
        selected = _select_cuda(device, output)
        _execute_checks(output, selected, checks)
        checks.update(status="passed", active_stage=None)
        _json_write(output / "checks.json", checks)
        return 0
    except BaseException as error:
        interrupted = isinstance(error, KeyboardInterrupt)
        checks["status"] = "interrupted" if interrupted else "failed"
        checks["failure"] = {"stage": checks["active_stage"], "type": type(error).__name__,
                             "code": getattr(error, "code", "gpu_check_failed")}
        if checks["active_stage"] in checks["stages"]:
            checks["stages"][checks["active_stage"]]["status"] = checks["status"]
        _json_write(output / "checks.json", checks)
        traceback.print_exc()
        return 130 if interrupted else 1


def _worker_command(output: Path, device: str) -> list[str]:
    return [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--device", device,
            "--output", str(output.resolve())]


def _supervise(output: Path, device: str) -> int:
    output.mkdir(parents=True, exist_ok=False)
    summary = {"schema_version": "wind3dgs.teacher_gpu_check_summary.v1", "status": "running", "started_at": _now(),
               "requested_device": device, "child_exit_code": None, "completed_stages": [], "last_active_stage": None,
               "failure": None, "convergence_status": "not_assessed"}
    _json_write(output / "summary.json", summary)
    started, child, interrupted = time.monotonic(), None, False
    with (output / "run.log").open("w", encoding="utf-8", buffering=1) as log:
        def emit(value):
            value = _redact(value, output)
            log.write(value)
            print(value, end="", flush=True)
        try:
            snapshot = _snapshot()
            snapshot["nvidia_smi"] = _nvidia_smi()
            # 외부 도구 오류에도 개인 경로가 남지 않도록 metadata까지 정규화한다.
            _json_write(output / "environment.json", json.loads(_redact(json.dumps(snapshot), output)))
            emit(f"Teacher GPU smoke 검사 시작: {device}\n")
            child = subprocess.Popen(_worker_command(output, device), cwd=CODE_ROOT, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", start_new_session=True)
            try:
                for line in child.stdout:
                    emit(line)
                child.wait()
            except KeyboardInterrupt:
                interrupted = True
                child.send_signal(signal.SIGINT)
                try:
                    remaining, _ = child.communicate(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    remaining, _ = child.communicate()
                emit(remaining or "")
            child.stdout.close()
            summary["child_exit_code"] = child.returncode
            checks_path = output / "checks.json"
            checks = json.loads(checks_path.read_text()) if checks_path.exists() else {}
            stages = checks.get("stages", {})
            summary["completed_stages"] = [name for name in STAGES if stages.get(name, {}).get("status") == "passed"]
            summary["last_active_stage"] = checks.get("active_stage")
            passed = child.returncode == 0 and checks.get("status") == "passed" and len(summary["completed_stages"]) == len(STAGES)
            summary["status"] = "interrupted" if interrupted else "passed" if passed else "failed"
            if not passed:
                summary["failure"] = checks.get("failure") or {"code": "worker_did_not_complete"}
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.kill()
                child.wait()
            summary["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            summary["failure"] = {"code": "supervisor_failed", "type": type(error).__name__}
            emit(traceback.format_exc())
        finally:
            summary.update(finished_at=_now(), elapsed_s=time.monotonic() - started)
            _json_write(output / "summary.json", summary)
            emit(f"검사 종료: {summary['status']} / 완료 단계 {len(summary['completed_stages'])}/{len(STAGES)}\n")
    return 0 if summary["status"] == "passed" else 130 if summary["status"] == "interrupted" else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CUDA Teacher 저장·probe·비교·재생 smoke 검사와 로그 저장")
    parser.add_argument("--device", default="cuda:0", help="실제 실행할 CUDA alias (기본: cuda:0)")
    parser.add_argument("--output", type=Path, help="새 결과 폴더. 상대 경로는 code 기준")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if re.fullmatch(r"cuda:[0-9]+", args.device) is None:
        parser.error("--device에는 cuda:0 같은 명시적 CUDA alias가 필요합니다")
    if args.worker:
        if args.output is None or not args.output.is_dir():
            parser.error("worker 출력 폴더가 필요합니다")
        return _worker(args.output, args.device)
    output = args.output or WORKSPACE / "experiments/artifacts/runs/teacher_gpu_check" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if not output.is_absolute():
        output = CODE_ROOT / output
    if output.exists():
        parser.error("결과 폴더가 이미 있습니다. 새 경로를 지정하세요")
    print(f"결과 폴더: {os.path.relpath(output, WORKSPACE)}", flush=True)
    return _supervise(output, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
