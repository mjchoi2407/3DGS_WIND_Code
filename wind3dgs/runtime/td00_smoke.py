"""TD00 거버넌스 smoke run과 compact 재현성 evidence를 생성한다."""

from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import wind3dgs

from wind3dgs.contracts import (
    RUNTIME_EXECUTION_ORDER,
    GlobalPrediction,
    PatchForceBatch,
    CorrectedDynamicState,
    RUN_MANIFEST_SCHEMA_VERSION,
    canonical_json_bytes,
    compute_reproducibility_key,
    load_run_manifest_schema,
    sha256_file,
    sha256_bytes,
    validate_run_manifest,
    verify_manifest_config,
    verify_manifest_outputs,
    write_canonical_json,
)
from wind3dgs.contracts.validation import ContractError


REPORT_SCHEMA_VERSION = "wind3dgs.td00_smoke_report.v1"
CONFIG_SCHEMA_VERSION = "wind3dgs.td00_smoke_config.v1"
TD00_PUBLICATION_MAX_FILE_BYTES = 1_048_576
TD00_REPOSITORY_PATHS = {
    "project": ".",
    "code": "code",
    "ideas": "ideas",
    "experiments": "experiments",
}
TD00_RUNTIME_PACKAGES = {
    "contracts",
    "io",
    "teacher",
    "topology",
    "aero",
    "reduced",
    "local",
    "learning",
    "runtime",
    "transport",
}
TD00_REQUIRED_CHECKS = {
    "manifest_schema_available",
    "manifest_schema_valid",
    "config_sha256_matches",
    "required_provenance_present",
    "source_revisions_clean",
    "runtime_execution_order_exact",
    "local_force_channels_separated",
    "cross_load_delta_separated",
    "rsh_mainline_dependency_absent",
    "legacy_completion_not_promoted",
    "artifact_policy_compliant",
    "output_paths_relative_and_contained",
    "output_sha256_matches",
    "no_absolute_path_or_secret",
    "governance_artifacts_present",
}
TD00_NOT_EVALUATED = {
    "mass_positive_definite": "TD01 E0",
    "damping_stiffness_psd": "TD01 E0",
    "global_local_mass_orthogonality": "TD01 E0",
    "force_torque_conservation": "TD01/TD11",
    "zero_wind_rest_invariance": "TD01/TD05",
    "affine_gaussian_transport": "TD01/TD13",
    "physics_rollout_accuracy": "TD05 and later",
    "typed_contract_completeness": "TD01",
    "end_to_end_force_ledger": "TD01 and TD11",
    "cross_delta_numeric_semantics": "TD01 and TD11",
    "local_base_aero_lineage": "TD01 and TD11",
    "json_schema_python_parity": "TD01 packaging/interop",
}
TD00_SCOPE = "governance, packaging, provenance, contract wiring만 검증"
_UNIX_ABSOLUTE_PATH_RE = re.compile(r"(?<![A-Za-z0-9/\\])/(?!/)[^\s'\"`\]),;]+")
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9/\\])(?:[A-Za-z]:[\\/]|\\\\[^\\/\s]+[\\/])"
)
_HOME_RELATIVE_PATH_RE = re.compile(r"(?<![A-Za-z0-9/\\])~(?:[\\/]|$)")


def validate_td00_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "milestone",
        "experiment_id",
        "seed",
        "device",
        "require_clean_sources",
        "inherit_legacy_completion",
        "artifact_run_root",
        "source_repositories",
        "dataset_id",
        "dataset_sha256_or_manifest_version",
        "object_package_id",
        "object_package_sha256",
        "models",
        "expected_stage_order",
        "mainline_packages",
        "publication_max_file_bytes",
    }
    missing = sorted(required - config.keys())
    extra = sorted(config.keys() - required)
    if missing or extra:
        raise ContractError(f"TD00 config field mismatch입니다. missing={missing}, extra={extra}")
    if config["schema_version"] != CONFIG_SCHEMA_VERSION or config["milestone"] != "TD00":
        raise ContractError("TD00 smoke config schema 또는 milestone이 예상과 다릅니다")
    if config["experiment_id"] != "TD00_contracts":
        raise ContractError("TD00 smoke experiment_id는 TD00_contracts여야 합니다")
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool) or config["seed"] < 0:
        raise ContractError("TD00 seed는 0 이상의 정수여야 합니다")
    if config["device"] != "cpu":
        raise ContractError("TD00 smoke는 CPU-only로 고정합니다")
    for field in ("require_clean_sources", "inherit_legacy_completion"):
        if not isinstance(config[field], bool):
            raise ContractError(f"{field}는 boolean이어야 합니다")
    if config["inherit_legacy_completion"]:
        raise ContractError("TD00은 legacy completion status를 승계할 수 없습니다")
    if config["models"] != []:
        raise ContractError("TD00 governance smoke에는 learned model을 선언할 수 없습니다")
    repositories = config["source_repositories"]
    if not isinstance(repositories, list) or not repositories:
        raise ContractError("source_repositories는 비어 있지 않은 list여야 합니다")
    actual_paths = {
        record.get("id"): record.get("relative_path")
        for record in repositories
        if isinstance(record, dict) and set(record) == {"id", "relative_path"}
    }
    if actual_paths != TD00_REPOSITORY_PATHS or len(repositories) != len(TD00_REPOSITORY_PATHS):
        raise ContractError("TD00 source repository ID/path가 split-repository contract와 일치해야 합니다")
    if not isinstance(config["mainline_packages"], list) or not all(
        isinstance(value, str) and value for value in config["mainline_packages"]
    ):
        raise ContractError("mainline_packages는 비어 있지 않은 string list여야 합니다")
    if set(config["mainline_packages"]) != TD00_RUNTIME_PACKAGES or len(config["mainline_packages"]) != len(
        TD00_RUNTIME_PACKAGES
    ):
        raise ContractError("mainline_packages에는 complete TD00 dependency-scan set을 정확히 한 번씩 넣어야 합니다")
    if not isinstance(config["expected_stage_order"], list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in config["expected_stage_order"]
    ):
        raise ContractError("expected_stage_order는 integer list여야 합니다")
    if config["publication_max_file_bytes"] != TD00_PUBLICATION_MAX_FILE_BYTES:
        raise ContractError("TD00 publication_max_file_bytes는 정확히 1048576이어야 합니다")


def discover_workspace(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / "manifests/artifact_policy.json").is_file() and (candidate / "code/wind3dgs").is_dir():
            return candidate
    raise ContractError("Wind3DGS workspace root를 찾지 못했습니다")


def _contained_path(workspace: Path, relative_path: str, name: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts:
        raise ContractError(f"{name}은 workspace 내부의 relative POSIX path여야 합니다")
    resolved = (workspace / pure).resolve()
    try:
        resolved.relative_to(workspace.resolve())
    except ValueError as exc:
        raise ContractError(f"{name}이 workspace 밖을 가리킵니다") from exc
    return resolved


def _git(worktree: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=worktree,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def snapshot_repositories(workspace: Path, records: list[dict[str, str]]) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for record in records:
        relative_path = record["relative_path"]
        worktree = workspace if relative_path == "." else _contained_path(workspace, relative_path, "repository path")
        git_toplevel = Path(_git(worktree, "rev-parse", "--show-toplevel")).resolve()
        if git_toplevel != worktree.resolve():
            raise ContractError(f"source repository path가 독립 Git worktree가 아닙니다: {relative_path}")
        commit = _git(worktree, "rev-parse", "HEAD")
        status = _git(worktree, "status", "--porcelain=v1", "--untracked-files=all")
        snapshots.append(
            {
                "id": record["id"],
                "relative_path": relative_path,
                "commit": commit,
                "dirty": bool(status),
            }
        )
    return snapshots


def require_unchanged_repository_snapshots(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
) -> None:
    if actual != expected:
        raise ContractError("TD00 실행 중 source repository revision 또는 dirty 상태가 변경되었습니다")


def artifact_run_root_is_ignored(workspace: Path, artifact_run_root: str) -> bool:
    experiments_root = workspace / "experiments"
    artifact_root = _contained_path(workspace, artifact_run_root, "artifact_run_root")
    try:
        probe = artifact_root.relative_to(experiments_root.resolve()) / ".td00-ignore-probe"
    except ValueError as exc:
        raise ContractError("artifact_run_root는 experiments repository 내부여야 합니다") from exc
    result = subprocess.run(
        ["git", "check-ignore", "--verbose", "--no-index", "--", probe.as_posix()],
        cwd=experiments_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise ContractError(f"git check-ignore 실행에 실패했습니다: {result.stderr.strip()}")
    if result.returncode != 0:
        return False
    source = result.stdout.split(":", 1)[0]
    return Path(source).name == ".gitignore"


def validate_executing_package(workspace: Path) -> None:
    expected = (workspace / "code/wind3dgs").resolve()
    actual = Path(wind3dgs.__file__).resolve().parent
    if actual != expected:
        raise ContractError(
            f"실행 중인 wind3dgs package가 기록 대상 code repo와 다릅니다: {actual} != {expected}"
        )


def governance_artifacts(workspace: Path) -> dict[str, bool]:
    required = {
        "experiment_template": workspace / "experiments/EXPERIMENT_TEMPLATE.md",
        "td00_readme": workspace / "experiments/TD00_contracts/README.md",
        "reuse_candidate_audit": workspace / "experiments/TD00_contracts/reports/reuse_candidate_audit.md",
    }
    return {name: path.is_file() and not path.is_symlink() for name, path in required.items()}


def scan_mainline_imports(code_root: Path, package_names: list[str]) -> list[str]:
    violations: list[str] = []
    source_groups: list[tuple[str, list[Path]]] = [("package_root", [code_root / "wind3dgs/__init__.py"])]
    for package_name in package_names:
        package_root = code_root / "wind3dgs" / package_name
        if not package_root.is_dir():
            violations.append(f"missing mainline package: {package_name}")
            continue
        source_groups.append((package_name, sorted(package_root.rglob("*.py"))))
    legacy_pattern = re.compile(r"^m[0-9]{2}(?:_|$)")
    allowed_wind3dgs_roots = set(package_names)

    def inspect_module_name(path: Path, module_name: str, *, dynamic: bool = False) -> None:
        if module_name.startswith("."):
            violations.append(
                f"{path.relative_to(code_root)} uses fail-closed relative dynamic import {module_name}"
            )
            return
        segments = module_name.lower().split(".")
        qualifier = "dynamic import" if dynamic else "import"
        if any(legacy_pattern.match(segment) for segment in segments):
            violations.append(f"{path.relative_to(code_root)} {qualifier}s legacy module {module_name}")
        if any(segment == "rsh" or segment.startswith("rsh_") for segment in segments):
            violations.append(f"{path.relative_to(code_root)} {qualifier}s RSH mainline module {module_name}")
        if segments[0] == "wind3dgs" and len(segments) > 1:
            imported_root = segments[1]
            if imported_root not in allowed_wind3dgs_roots:
                violations.append(
                    f"{path.relative_to(code_root)} imports unscanned wind3dgs module {module_name}"
                )

    for group_name, paths in source_groups:
        for path in paths:
            if not path.is_file():
                violations.append(f"missing mainline source: {path.relative_to(code_root)}")
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                module_names: list[str] = []
                if isinstance(node, ast.Import):
                    module_names.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        for alias in node.names:
                            relative_target = node.module or alias.name
                            if node.level == 1 and group_name != "package_root":
                                resolved_relative = f"wind3dgs.{group_name}.{relative_target}"
                            elif node.level == 2:
                                resolved_relative = f"wind3dgs.{relative_target}"
                            else:
                                violations.append(
                                    f"{path.relative_to(code_root)} uses unsupported relative import {relative_target}"
                                )
                                continue
                            inspect_module_name(path, resolved_relative)
                    else:
                        if node.module:
                            module_names.append(node.module)
                            module_names.extend(f"{node.module}.{alias.name}" for alias in node.names)
                        else:
                            module_names.extend(alias.name for alias in node.names)
                for module_name in module_names:
                    inspect_module_name(path, module_name)
                if isinstance(node, ast.Call):
                    function_name = ""
                    if isinstance(node.func, ast.Name):
                        function_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        function_name = node.func.attr
                    if function_name in {"import_module", "__import__"}:
                        if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(
                            node.args[0].value, str
                        ):
                            violations.append(
                                f"{path.relative_to(code_root)} uses non-literal dynamic import"
                            )
                        else:
                            inspect_module_name(path, node.args[0].value, dynamic=True)
    return violations


def _check(status: str, message: str, evidence: Any) -> dict[str, Any]:
    return {"status": status, "message": message, "evidence": evidence}


def _assert_no_private_or_absolute_strings(payload: Any, path: str = "root") -> None:
    secret_keys = {
        "access_token",
        "api_key",
        "authorization",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "token",
    }
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key)
            if key_text.lower() in secret_keys:
                raise ContractError(f"publication에 secret-like key가 있습니다: {path}.{key}")
            _assert_no_private_or_absolute_strings(key_text, f"{path}.<key>")
            _assert_no_private_or_absolute_strings(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _assert_no_private_or_absolute_strings(value, f"{path}[{index}]")
    elif isinstance(payload, str):
        if (
            _UNIX_ABSOLUTE_PATH_RE.search(payload)
            or _WINDOWS_ABSOLUTE_PATH_RE.search(payload)
            or _HOME_RELATIVE_PATH_RE.search(payload)
        ):
            raise ContractError(f"publication에 absolute/private path가 있습니다: {path}")


def _stable_command(config_relative: str, publish_relative: str | None) -> list[str]:
    command = [".venv/bin/python", "-m", "wind3dgs.runtime.td00_smoke", "--config", config_relative]
    if publish_relative:
        command.extend(("--publish-dir", publish_relative))
    return command


def _build_report(
    *,
    run_id: str,
    config: dict[str, Any],
    repositories: list[dict[str, Any]],
    import_violations: list[str],
    workspace: Path,
    config_sha256: str,
    artifact_root_ignored: bool,
) -> dict[str, Any]:
    expected_order = [int(stage) for stage in RUNTIME_EXECUTION_ORDER]
    declared_order = config["expected_stage_order"]
    patch_fields = {item.name for item in fields(PatchForceBatch)}
    global_fields = {item.name for item in fields(GlobalPrediction)}
    corrected_fields = {item.name for item in fields(CorrectedDynamicState)}
    policy = json.loads((workspace / "manifests/artifact_policy.json").read_text(encoding="utf-8"))
    code_readme = (workspace / "code/README.md").read_text(encoding="utf-8")
    experiment_readme = (workspace / "experiments/README.md").read_text(encoding="utf-8")
    governance_files = governance_artifacts(workspace)
    checks = {
        "manifest_schema_available": _check(
            "pass",
            "배포 package에서 JSON Schema resource를 읽을 수 있습니다.",
            load_run_manifest_schema()["$id"],
        ),
        "manifest_schema_valid": _check(
            "pass",
            "최종 manifest는 Python 기준 validator를 통과한 뒤 발행됩니다.",
            RUN_MANIFEST_SCHEMA_VERSION,
        ),
        "config_sha256_matches": _check(
            "pass",
            "실행에 사용한 config byte hash를 최종 manifest에 기록합니다.",
            config_sha256,
        ),
        "required_provenance_present": _check(
            "pass",
            "분리된 네 source repository의 revision을 모두 확인했습니다.",
            [record["id"] for record in repositories],
        ),
        "source_revisions_clean": _check(
            "pass" if not any(record["dirty"] for record in repositories) else "fail",
            "reference smoke 실행 전 모든 source worktree가 clean이어야 합니다.",
            {record["id"]: record["dirty"] for record in repositories},
        ),
        "runtime_execution_order_exact": _check(
            "pass" if declared_order == expected_order else "fail",
            "stage 10 조립과 stage 9 complement를 거친 뒤 stage 8 Local solve를 한 번 수행합니다.",
            declared_order,
        ),
        "local_force_channels_separated": _check(
            "pass"
            if {"analytic_base_aero", "learned_missing_aero", "learned_missing_structural"} <= patch_fields
            else "fail",
            "Local analytic aero와 두 learned missing-force channel을 분리해 유지합니다.",
            sorted(patch_fields),
        ),
        "cross_load_delta_separated": _check(
            "pass"
            if "predictor_cross_load" in global_fields and "structural_cross_delta" in corrected_fields
            else "fail",
            "Global predictor가 이미 소비한 cross load와 corrector delta를 따로 기록합니다.",
            {
                "global_prediction": sorted(global_fields),
                "corrected_dynamic_state": sorted(corrected_fields),
            },
        ),
        "rsh_mainline_dependency_absent": _check(
            "pass" if not import_violations else "fail",
            "TD runtime package는 legacy M module이나 RSH runtime module을 import하지 않습니다.",
            import_violations,
        ),
        "legacy_completion_not_promoted": _check(
            "pass"
            if not config["inherit_legacy_completion"]
            and "td00:legacy-results-not-evidence" in code_readme
            and "td00:legacy-status-not-inherited" in experiment_readme
            else "fail",
            "M01--M04 결과는 재사용 후보, baseline, support로만 유지합니다.",
            {"inherit_legacy_completion": config["inherit_legacy_completion"]},
        ),
        "artifact_policy_compliant": _check(
            "pass"
            if policy["physical_layout"]["shared_heavy_root"] == "experiments/artifacts/"
            and config["artifact_run_root"].startswith("experiments/artifacts/")
            and artifact_root_ignored
            else "fail",
            "working run은 Git에서 제외된 heavy-artifact root 아래에 저장합니다.",
            {"path": config["artifact_run_root"], "git_ignored": artifact_root_ignored},
        ),
        "output_paths_relative_and_contained": _check(
            "pass",
            "run과 publication 경로는 선언된 workspace 내부로 제한합니다.",
            [config["artifact_run_root"], "experiments/TD00_contracts/reports"],
        ),
        "output_sha256_matches": _check(
            "pass",
            "원자적 run-directory rename 전후에 output 크기와 SHA-256을 검증합니다.",
            "enforced_by_generator",
        ),
        "no_absolute_path_or_secret": _check(
            "pass",
            "compact evidence는 발행 전에 절대경로와 secret 유사 항목을 검사합니다.",
            "enforced_by_generator",
        ),
        "governance_artifacts_present": _check(
            "pass" if all(governance_files.values()) else "fail",
            "TD00 필수 README, 공통 template, 재사용 감사가 존재합니다.",
            governance_files,
        ),
    }
    status = "pass" if all(item["status"] == "pass" for item in checks.values()) else "fail"
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "milestone": "TD00",
        "status": status,
        "scope": TD00_SCOPE,
        "checks": checks,
        "not_evaluated": TD00_NOT_EVALUATED,
    }


def _publish_compact_evidence(
    run_directory: Path,
    publish_directory: Path,
    max_file_bytes: int,
) -> None:
    if max_file_bytes != TD00_PUBLICATION_MAX_FILE_BYTES:
        raise ContractError("TD00 compact evidence limit은 정확히 1 MiB여야 합니다")
    publish_directory.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[Path, Path, bytes]] = []
    payloads: dict[str, dict[str, Any]] = {}
    completion_target = publish_directory / "reference_complete.json"
    completion_staging = publish_directory / ".reference_complete.json.td00.tmp"
    if completion_target.is_symlink() or completion_staging.exists() or completion_staging.is_symlink():
        raise ContractError("TD00 completion marker path가 안전하지 않습니다")
    for filename in ("run_manifest.json", "td00_smoke_report.json"):
        source = run_directory / filename
        source_bytes = source.read_bytes()
        if len(source_bytes) > max_file_bytes:
            raise ContractError(f"reference evidence가 publication limit을 초과합니다: {filename}")
        payload = json.loads(source_bytes.decode("utf-8"))
        payloads[filename] = payload
        _assert_no_private_or_absolute_strings(payload)
        target = publish_directory / filename
        if target.is_symlink():
            raise ContractError(f"symlink를 통한 publication을 거부합니다: {target.name}")
        if target.exists() and target.read_bytes() != source_bytes:
            raise ContractError(f"내용이 다른 reference evidence를 덮어쓰지 않습니다: {target.name}")
        staging = target.with_name(f".{target.name}.td00.tmp")
        if staging.exists() or staging.is_symlink():
            raise ContractError(f"stale TD00 publication staging file이 남아 있습니다: {staging.name}")
        pending.append((target, staging, source_bytes))
    _verify_td00_report_cross_fields(
        payloads["run_manifest.json"], payloads["td00_smoke_report.json"]
    )
    manifest_bytes = next(
        source_bytes for target, _staging, source_bytes in pending if target.name == "run_manifest.json"
    )
    report_bytes = next(
        source_bytes for target, _staging, source_bytes in pending if target.name == "td00_smoke_report.json"
    )
    completion = {
        "schema_version": "wind3dgs.td00_reference_complete.v1",
        "run_id": payloads["run_manifest.json"]["run_id"],
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "report_sha256": sha256_bytes(report_bytes),
    }
    completion_bytes = canonical_json_bytes(completion)
    if completion_target.exists() and completion_target.read_bytes() != completion_bytes:
        raise ContractError("기존 TD00 completion marker와 새 evidence가 다릅니다")
    for _target, staging, source_bytes in pending:
        staging.write_bytes(source_bytes)
    completion_staging.write_bytes(completion_bytes)
    for target, staging, _source_bytes in pending:
        os.replace(staging, target)
    os.replace(completion_staging, completion_target)
    verify_td00_evidence_directory(publish_directory)


def _verify_td00_report_cross_fields(manifest: dict[str, Any], report: dict[str, Any]) -> None:
    required_report_fields = {
        "schema_version",
        "run_id",
        "milestone",
        "status",
        "scope",
        "checks",
        "not_evaluated",
    }
    if set(report) != required_report_fields:
        raise ContractError("TD00 report는 versioned top-level field 집합과 정확히 일치해야 합니다")
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ContractError("TD00 report의 schema_version을 지원하지 않습니다")
    for field in ("run_id", "milestone"):
        if report.get(field) != manifest.get(field):
            raise ContractError(f"TD00 report의 {field}가 manifest와 일치하지 않습니다")
    if report.get("status") != "pass":
        raise ContractError("TD00 reference report status는 pass여야 합니다")
    checks = report.get("checks")
    if not isinstance(checks, dict) or set(checks) != TD00_REQUIRED_CHECKS or any(
        not isinstance(check, dict)
        or set(check) != {"status", "message", "evidence"}
        or check.get("status") != "pass"
        or not isinstance(check.get("message"), str)
        or not check["message"].strip()
        for check in checks.values()
    ):
        raise ContractError("TD00 reference report는 필수 passing check를 정확히 포함해야 합니다")
    if report.get("scope") != TD00_SCOPE:
        raise ContractError("TD00 report scope가 없거나 잘못되었습니다")
    if report.get("not_evaluated") != TD00_NOT_EVALUATED:
        raise ContractError("TD00 report는 미검증 TD01+ check 집합을 정확히 공개해야 합니다")
    output_records = {record["id"]: record for record in manifest.get("outputs", [])}
    report_record = output_records.get("td00_smoke_report")
    if report_record is None or report_record.get("path") != "td00_smoke_report.json":
        raise ContractError("manifest가 td00_smoke_report.json을 TD00 evidence로 식별해야 합니다")


def validate_td00_manifest_semantics(manifest: dict[str, Any]) -> None:
    if manifest.get("milestone") != "TD00":
        raise ContractError("TD00 evidence manifest의 milestone은 TD00이어야 합니다")
    repository_paths = {
        record.get("id"): record.get("relative_path")
        for record in manifest.get("source_repositories", [])
        if isinstance(record, dict)
    }
    if repository_paths != TD00_REPOSITORY_PATHS or len(manifest.get("source_repositories", [])) != 4:
        raise ContractError("TD00 evidence는 분리된 네 repository를 정확히 기록해야 합니다")
    if any(record.get("dirty") for record in manifest["source_repositories"]):
        raise ContractError("dirty source에서는 TD00 reference evidence를 만들 수 없습니다")
    if manifest.get("device") != {
        "requested": "cpu",
        "resolved": "cpu",
        "class": "cpu-contract-smoke",
    }:
        raise ContractError("TD00 evidence는 CPU contract-smoke device를 사용해야 합니다")
    if manifest.get("working_directory") != "." or manifest.get("environment") != {"PYTHONPATH": "code"}:
        raise ContractError("TD00 evidence의 working directory 또는 environment가 canonical하지 않습니다")
    canonical_config = "code/configs/td00_contracts_smoke.json"
    command = manifest.get("command", [])
    base_command = _stable_command(canonical_config, None)
    command_is_canonical = command == base_command
    if len(command) == len(base_command) + 2 and command[: len(base_command)] == base_command:
        publish_path = PurePosixPath(command[-1])
        command_is_canonical = (
            command[-2] == "--publish-dir"
            and not publish_path.is_absolute()
            and ".." not in publish_path.parts
            and publish_path.parts[:3] == ("experiments", "TD00_contracts", "reports")
            and len(publish_path.parts) > 3
        )
    if not command_is_canonical:
        raise ContractError("TD00 evidence command가 canonical argument 형태와 정확히 일치하지 않습니다")
    if manifest.get("config_path") != canonical_config:
        raise ContractError("TD00 evidence는 canonical smoke config를 사용해야 합니다")
    outputs = manifest.get("outputs", [])
    if (
        len(outputs) != 1
        or outputs[0].get("id") != "td00_smoke_report"
        or outputs[0].get("path") != "td00_smoke_report.json"
    ):
        raise ContractError("TD00 evidence manifest는 smoke report output 하나만 기록해야 합니다")


def verify_td00_evidence_directory(
    evidence_directory: Path,
    *,
    max_file_bytes: int = TD00_PUBLICATION_MAX_FILE_BYTES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a complete on-disk TD00 manifest/report pair."""

    if max_file_bytes != TD00_PUBLICATION_MAX_FILE_BYTES:
        raise ContractError("TD00 evidence verifier limit은 정확히 1 MiB여야 합니다")
    evidence_directory = evidence_directory.resolve()
    manifest_path = evidence_directory / "run_manifest.json"
    report_path = evidence_directory / "td00_smoke_report.json"
    completion_path = evidence_directory / "reference_complete.json"
    if manifest_path.is_symlink() or report_path.is_symlink() or completion_path.is_symlink():
        raise ContractError("TD00 evidence file은 symlink일 수 없습니다")
    try:
        manifest_bytes = manifest_path.read_bytes()
        report_bytes = report_path.read_bytes()
        completion_bytes = completion_path.read_bytes()
    except FileNotFoundError as exc:
        raise ContractError("TD00 evidence directory에 manifest/report pair가 없습니다") from exc
    for filename, payload_bytes in (
        (manifest_path.name, manifest_bytes),
        (report_path.name, report_bytes),
        (completion_path.name, completion_bytes),
    ):
        if len(payload_bytes) > max_file_bytes:
            raise ContractError(f"TD00 evidence가 compact publication limit을 초과합니다: {filename}")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    report = json.loads(report_bytes.decode("utf-8"))
    completion = json.loads(completion_bytes.decode("utf-8"))
    validate_run_manifest(manifest)
    validate_td00_manifest_semantics(manifest)
    _verify_td00_report_cross_fields(manifest, report)
    _assert_no_private_or_absolute_strings(manifest)
    _assert_no_private_or_absolute_strings(report)
    _assert_no_private_or_absolute_strings(completion)
    output_records = {record["id"]: record for record in manifest["outputs"]}
    report_record = output_records["td00_smoke_report"]
    if report_record["size_bytes"] != len(report_bytes) or report_record["sha256"] != sha256_bytes(report_bytes):
        raise ContractError("TD00 report bytes가 manifest output record와 일치하지 않습니다")
    expected_completion = {
        "schema_version": "wind3dgs.td00_reference_complete.v1",
        "run_id": manifest["run_id"],
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "report_sha256": sha256_bytes(report_bytes),
    }
    if completion != expected_completion:
        raise ContractError("TD00 reference completion marker가 evidence pair와 일치하지 않습니다")
    return manifest, report


def run_smoke(
    *,
    workspace: Path,
    config_relative: str,
    publish_relative: str | None,
) -> tuple[Path, dict[str, Any]]:
    config_path = _contained_path(workspace, config_relative, "config")
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes.decode("utf-8"))
    validate_td00_config(config)
    validate_executing_package(workspace)
    config_sha256 = sha256_bytes(config_bytes)
    repositories = snapshot_repositories(workspace, config["source_repositories"])
    if config["require_clean_sources"] and any(record["dirty"] for record in repositories):
        dirty = [record["id"] for record in repositories if record["dirty"]]
        raise ContractError(f"reference smoke에는 clean source repository가 필요합니다. dirty={', '.join(dirty)}")
    code_root = workspace / "code"
    import_violations = scan_mainline_imports(code_root, config["mainline_packages"])
    artifact_root_ignored = artifact_run_root_is_ignored(workspace, config["artifact_run_root"])
    if not artifact_root_ignored:
        raise ContractError("artifact run root가 version-controlled experiments/.gitignore에 포함되지 않았습니다")
    if not all(governance_artifacts(workspace).values()):
        raise ContractError("TD00 필수 governance 문서가 누락되었습니다")
    publish_directory: Path | None = None
    if publish_relative is not None:
        publish_directory = _contained_path(workspace, publish_relative, "publish directory")
        expected_parent = (workspace / "experiments/TD00_contracts/reports").resolve()
        try:
            publish_directory.relative_to(expected_parent)
        except ValueError as exc:
            raise ContractError(
                "TD00 compact evidence는 experiments/TD00_contracts/reports 아래에 발행해야 합니다"
            ) from exc

    code_commit = next(record["commit"] for record in repositories if record["id"] == "code")
    created_at = datetime.now(timezone.utc)
    run_id = f"td00-contracts-smoke-{created_at.strftime('%Y%m%dT%H%M%S%fZ').lower()}-{code_commit[:7]}"
    report = _build_report(
        run_id=run_id,
        config=config,
        repositories=repositories,
        import_violations=import_violations,
        workspace=workspace,
        config_sha256=config_sha256,
        artifact_root_ignored=artifact_root_ignored,
    )
    if report["status"] != "pass":
        failed_checks = sorted(key for key, value in report["checks"].items() if value["status"] != "pass")
        raise ContractError(f"TD00 smoke check 실패: {', '.join(failed_checks)}")
    report_bytes = canonical_json_bytes(report)

    command = _stable_command(config_relative, publish_relative)
    reproducibility_input = {
        "command": command,
        "working_directory": ".",
        "environment": {"PYTHONPATH": "code"},
        "config_sha256": config_sha256,
        "dataset_id": config["dataset_id"],
        "dataset_sha256_or_manifest_version": config["dataset_sha256_or_manifest_version"],
        "device": {"requested": "cpu", "resolved": "cpu", "class": "cpu-contract-smoke"},
        "milestone": config["milestone"],
        "object_package_id": config["object_package_id"],
        "object_package_sha256": config["object_package_sha256"],
        "models": config["models"],
        "seed": config["seed"],
        "source_repositories": [
            {"id": record["id"], "commit": record["commit"]} for record in repositories
        ],
    }
    reproducibility_key = compute_reproducibility_key(reproducibility_input)
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "milestone": config["milestone"],
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
        "source_repositories": repositories,
        "command": command,
        "working_directory": ".",
        "environment": {"PYTHONPATH": "code"},
        "config_path": config_relative,
        "config_sha256": config_sha256,
        "seed": config["seed"],
        "device": {"requested": "cpu", "resolved": "cpu", "class": "cpu-contract-smoke"},
        "dataset_id": config["dataset_id"],
        "dataset_sha256_or_manifest_version": config["dataset_sha256_or_manifest_version"],
        "object_package_id": config["object_package_id"],
        "object_package_sha256": config["object_package_sha256"],
        "models": config["models"],
        "outputs": [
            {
                "id": "td00_smoke_report",
                "path": "td00_smoke_report.json",
                "sha256": sha256_bytes(report_bytes),
                "size_bytes": len(report_bytes),
            }
        ],
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "reproducibility_key": reproducibility_key,
    }
    validate_run_manifest(manifest)
    validate_td00_manifest_semantics(manifest)
    _verify_td00_report_cross_fields(manifest, report)
    verify_manifest_config(manifest, workspace)
    require_unchanged_repository_snapshots(
        repositories,
        snapshot_repositories(workspace, config["source_repositories"]),
    )
    _assert_no_private_or_absolute_strings(manifest)
    _assert_no_private_or_absolute_strings(report)

    run_root = _contained_path(workspace, config["artifact_run_root"], "artifact_run_root")
    run_root.mkdir(parents=True, exist_ok=True)
    temporary_directory = run_root / f".{run_id}.tmp"
    final_directory = run_root / run_id
    if temporary_directory.exists() or final_directory.exists():
        raise ContractError(f"run directory 이름이 충돌합니다: {run_id}")
    temporary_directory.mkdir()
    (temporary_directory / "td00_smoke_report.json").write_bytes(report_bytes)
    write_canonical_json(temporary_directory / "run_manifest.json", manifest)
    verify_manifest_outputs(manifest, temporary_directory)
    require_unchanged_repository_snapshots(
        repositories,
        snapshot_repositories(workspace, config["source_repositories"]),
    )
    temporary_directory.rename(final_directory)
    verify_manifest_outputs(manifest, final_directory)

    if publish_directory is not None:
        _publish_compact_evidence(final_directory, publish_directory, config["publication_max_file_bytes"])

    return final_directory, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="code/configs/td00_contracts_smoke.json")
    parser.add_argument("--publish-dir")
    parser.add_argument("--workspace-root", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace = args.workspace_root.resolve() if args.workspace_root else discover_workspace(Path.cwd())
    try:
        run_directory, manifest = run_smoke(
            workspace=workspace,
            config_relative=args.config,
            publish_relative=args.publish_dir,
        )
    except (ContractError, OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(
            f"TD00 smoke 실패: 계약 또는 재현성 검증을 통과하지 못했습니다. 상세={exc}",
            file=sys.stderr,
        )
        return 1
    relative_run = run_directory.relative_to(workspace)
    print(
        json.dumps(
            {
                "status": "pass",
                "run_id": manifest["run_id"],
                "run_directory": relative_run.as_posix(),
                "reproducibility_key": manifest["reproducibility_key"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
