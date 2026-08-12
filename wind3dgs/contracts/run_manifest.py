"""Versioned run-manifest serialization and validation helpers."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any

from .validation import ContractError, require_nonempty, require_sha256


RUN_MANIFEST_SCHEMA_VERSION = "wind3dgs.run_manifest.v1"
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
RFC3339_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]+)?(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)
TD00_DATASET_ID = "not_applicable:td00_governance_smoke"
TD00_OBJECT_PACKAGE_ID = "not_applicable:td00_governance_smoke"


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_canonical_json(path: Path, payload: Any) -> None:
    path.write_bytes(canonical_json_bytes(payload))


def reproducibility_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the stable logical inputs covered by reproducibility_key."""

    return {
        "command": payload["command"],
        "working_directory": payload["working_directory"],
        "environment": payload["environment"],
        "config_sha256": payload["config_sha256"],
        "dataset_id": payload["dataset_id"],
        "dataset_sha256_or_manifest_version": payload["dataset_sha256_or_manifest_version"],
        "device": payload["device"]["requested"] if isinstance(payload["device"], dict) else payload["device"],
        "milestone": payload["milestone"],
        "object_package_id": payload["object_package_id"],
        "object_package_sha256": payload["object_package_sha256"],
        "models": payload["models"],
        "seed": payload["seed"],
        "source_repositories": [
            {"id": record["id"], "commit": record["commit"]}
            for record in payload["source_repositories"]
        ],
    }


def compute_reproducibility_key(payload: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(reproducibility_payload(payload)))


def load_run_manifest_schema() -> dict[str, Any]:
    schema_resource = resources.files("wind3dgs.contracts").joinpath("schemas/run_manifest.schema.json")
    return json.loads(schema_resource.read_text(encoding="utf-8"))


def _require_relative_path(value: str, name: str) -> None:
    require_nonempty(value, name)
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "" in path.parts:
        raise ContractError(f"{name} must be a contained relative POSIX path")


def validate_run_manifest(payload: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "run_id",
        "milestone",
        "created_at",
        "source_repositories",
        "command",
        "working_directory",
        "environment",
        "config_path",
        "config_sha256",
        "seed",
        "device",
        "dataset_id",
        "dataset_sha256_or_manifest_version",
        "object_package_id",
        "object_package_sha256",
        "models",
        "outputs",
        "software",
        "reproducibility_key",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ContractError(f"run manifest is missing required fields: {', '.join(missing)}")
    extra = sorted(payload.keys() - required)
    if extra:
        raise ContractError(f"run manifest has unknown fields: {', '.join(extra)}")
    if payload["schema_version"] != RUN_MANIFEST_SCHEMA_VERSION:
        raise ContractError(f"unsupported run manifest schema: {payload['schema_version']}")
    run_id = payload["run_id"]
    if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
        raise ContractError("run_id has an invalid format")
    if not isinstance(payload["milestone"], str) or not re.fullmatch(r"TD[0-9]{2}", payload["milestone"]):
        raise ContractError("milestone must use the TD## namespace")
    created_at = payload["created_at"]
    if not isinstance(created_at, str) or not RFC3339_RE.fullmatch(created_at):
        raise ContractError("created_at must be an RFC-3339 date-time with timezone")
    try:
        parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError("created_at must be a valid RFC-3339 date-time") from exc
    if parsed_created_at.tzinfo is None:
        raise ContractError("created_at must include a timezone")
    if not isinstance(payload["seed"], int) or isinstance(payload["seed"], bool) or payload["seed"] < 0:
        raise ContractError("seed must be a nonnegative integer")
    _require_relative_path(payload["config_path"], "config_path")
    require_sha256(payload["config_sha256"], "config_sha256")
    require_sha256(payload["reproducibility_key"], "reproducibility_key")
    command = payload["command"]
    if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
        raise ContractError("command must be a non-empty string list")
    if "--config" not in command:
        raise ContractError("command must include --config")
    config_argument_index = command.index("--config") + 1
    if config_argument_index >= len(command) or command[config_argument_index] != payload["config_path"]:
        raise ContractError("command --config value must match config_path")
    working_directory = payload["working_directory"]
    if working_directory != ".":
        _require_relative_path(working_directory, "working_directory")
    environment = payload["environment"]
    if not isinstance(environment, dict):
        raise ContractError("environment must be an object")
    if any(not isinstance(key, str) or not key or not isinstance(value, str) for key, value in environment.items()):
        raise ContractError("environment keys and values must be strings")
    if set(environment) - {"PYTHONPATH"}:
        raise ContractError("run manifest environment may record only approved reproducibility variables")
    for key, value in environment.items():
        if value != ".":
            _require_relative_path(value, f"environment.{key}")
    repositories = payload["source_repositories"]
    if not isinstance(repositories, list) or not repositories:
        raise ContractError("source_repositories must be a non-empty list")
    repository_ids: set[str] = set()
    for repository in repositories:
        if not isinstance(repository, dict):
            raise ContractError("source repository records must be objects")
        for field in ("id", "relative_path", "commit", "dirty"):
            if field not in repository:
                raise ContractError(f"source repository record is missing {field}")
        extra_repository_fields = set(repository) - {"id", "relative_path", "commit", "dirty"}
        if extra_repository_fields:
            raise ContractError("source repository record has unknown fields")
        require_nonempty(repository["id"], "source repository id")
        if repository["id"] in repository_ids:
            raise ContractError(f"duplicate source repository id: {repository['id']}")
        repository_ids.add(repository["id"])
        if repository["relative_path"] != ".":
            _require_relative_path(repository["relative_path"], "source repository relative_path")
        if not COMMIT_RE.fullmatch(repository["commit"]):
            raise ContractError("source repository commit must be a full lowercase Git SHA")
        if not isinstance(repository["dirty"], bool):
            raise ContractError("source repository dirty must be boolean")
    device = payload["device"]
    if not isinstance(device, dict) or set(("requested", "resolved", "class")) - device.keys():
        raise ContractError("device must contain requested, resolved, and class")
    if set(device) != {"requested", "resolved", "class"}:
        raise ContractError("device has unknown fields")
    for key, value in device.items():
        require_nonempty(value, f"device.{key}")
    for field in ("dataset_id", "dataset_sha256_or_manifest_version", "object_package_id", "object_package_sha256"):
        require_nonempty(payload[field], field)
    if payload["milestone"] == "TD00":
        if payload["dataset_id"] != TD00_DATASET_ID:
            raise ContractError("TD00 must use the explicit governance-smoke dataset sentinel")
        if payload["object_package_id"] != TD00_OBJECT_PACKAGE_ID:
            raise ContractError("TD00 must use the explicit governance-smoke object-package sentinel")
        if payload["object_package_sha256"] != TD00_OBJECT_PACKAGE_ID:
            raise ContractError("TD00 must use the explicit governance-smoke object-package hash sentinel")
    else:
        if str(payload["dataset_id"]).startswith("not_applicable:"):
            raise ContractError("dataset not_applicable sentinel is allowed only for TD00")
        if str(payload["object_package_id"]).startswith("not_applicable:"):
            raise ContractError("object package not_applicable sentinel is allowed only for TD00")
        require_sha256(payload["object_package_sha256"], "object_package_sha256")
    models = payload["models"]
    if not isinstance(models, list):
        raise ContractError("models must be a list")
    if payload["milestone"] == "TD00" and models:
        raise ContractError("TD00 governance smoke cannot declare learned models")
    model_ids: set[str] = set()
    for model in models:
        if not isinstance(model, dict) or set(model) != {"id", "sha256"}:
            raise ContractError("model records must contain exactly id and sha256")
        require_nonempty(model["id"], "model id")
        require_sha256(model["sha256"], "model sha256")
        if model["id"] in model_ids:
            raise ContractError(f"duplicate model id: {model['id']}")
        model_ids.add(model["id"])
    outputs = payload["outputs"]
    if not isinstance(outputs, list) or not outputs:
        raise ContractError("outputs must be a non-empty list")
    output_ids: set[str] = set()
    for output in outputs:
        if not isinstance(output, dict):
            raise ContractError("output records must be objects")
        if set(("id", "path", "sha256", "size_bytes")) - output.keys():
            raise ContractError("output records need id, path, sha256, and size_bytes")
        if set(output) != {"id", "path", "sha256", "size_bytes"}:
            raise ContractError("output record has unknown fields")
        require_nonempty(output["id"], "output id")
        if output["id"] in output_ids:
            raise ContractError(f"duplicate output id: {output['id']}")
        output_ids.add(output["id"])
        _require_relative_path(output["path"], "output path")
        require_sha256(output["sha256"], "output sha256")
        if (
            not isinstance(output["size_bytes"], int)
            or isinstance(output["size_bytes"], bool)
            or output["size_bytes"] < 0
        ):
            raise ContractError("output size_bytes must be a nonnegative integer")
    software = payload["software"]
    if not isinstance(software, dict) or set(software) != {"python", "numpy", "platform"}:
        raise ContractError("software must contain exactly python, numpy, and platform")
    for key, value in software.items():
        require_nonempty(value, f"software.{key}")
    expected_reproducibility_key = compute_reproducibility_key(payload)
    if payload["reproducibility_key"] != expected_reproducibility_key:
        raise ContractError("reproducibility_key does not match the logical run inputs")


def verify_manifest_outputs(payload: dict[str, Any], run_directory: Path) -> None:
    run_directory = run_directory.resolve()
    for output in payload["outputs"]:
        relative = PurePosixPath(output["path"])
        lexical_candidate = run_directory
        for part in relative.parts:
            lexical_candidate = lexical_candidate / part
            if lexical_candidate.is_symlink():
                raise ContractError(f"output path traverses a symlink: {output['path']}")
        candidate = lexical_candidate.resolve()
        try:
            candidate.relative_to(run_directory)
        except ValueError as exc:
            raise ContractError(f"output escapes run directory: {output['path']}") from exc
        if not candidate.is_file():
            raise ContractError(f"output is missing or is not a regular file: {output['path']}")
        if candidate.stat().st_size != output["size_bytes"]:
            raise ContractError(f"output size mismatch: {output['path']}")
        if sha256_file(candidate) != output["sha256"]:
            raise ContractError(f"output SHA-256 mismatch: {output['path']}")


def verify_manifest_config(payload: dict[str, Any], workspace: Path) -> None:
    workspace = workspace.resolve()
    relative = PurePosixPath(payload["config_path"])
    lexical_path = workspace
    for part in relative.parts:
        lexical_path = lexical_path / part
        if lexical_path.is_symlink():
            raise ContractError("config path traverses a symlink")
    config_path = lexical_path.resolve()
    try:
        config_path.relative_to(workspace)
    except ValueError as exc:
        raise ContractError("config path escapes the workspace") from exc
    if not config_path.is_file():
        raise ContractError("config path is missing")
    if sha256_file(config_path) != payload["config_sha256"]:
        raise ContractError("config SHA-256 does not match the recorded manifest value")
