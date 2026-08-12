from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from wind3dgs.contracts import (
    ContractError,
    RUN_MANIFEST_SCHEMA_VERSION,
    load_run_manifest_schema,
    compute_reproducibility_key,
    sha256_file,
    validate_run_manifest,
    verify_manifest_outputs,
    verify_manifest_config,
    write_canonical_json,
)


def valid_manifest(report_path: Path) -> dict:
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_id": "td00-test-run",
        "milestone": "TD00",
        "created_at": "2026-08-13T00:00:00Z",
        "source_repositories": [
            {"id": "code", "relative_path": "code", "commit": "a" * 40, "dirty": False}
        ],
        "command": [
            "python",
            "-m",
            "wind3dgs.runtime.td00_smoke",
            "--config",
            "code/configs/td00_contracts_smoke.json",
        ],
        "working_directory": ".",
        "environment": {"PYTHONPATH": "code"},
        "config_path": "code/configs/td00_contracts_smoke.json",
        "config_sha256": "b" * 64,
        "seed": 7,
        "device": {"requested": "cpu", "resolved": "cpu", "class": "cpu-contract-smoke"},
        "dataset_id": "not_applicable:td00_governance_smoke",
        "dataset_sha256_or_manifest_version": "manifest:manifests/datasets.json@1.0.0",
        "object_package_id": "not_applicable:td00_governance_smoke",
        "object_package_sha256": "not_applicable:td00_governance_smoke",
        "models": [],
        "outputs": [
            {
                "id": "report",
                "path": report_path.name,
                "sha256": sha256_file(report_path),
                "size_bytes": report_path.stat().st_size,
            }
        ],
        "software": {"python": "3.12.3", "numpy": "2.4.4", "platform": "test"},
        "reproducibility_key": "0" * 64,
    }


def finalized_manifest(report_path: Path) -> dict:
    manifest = valid_manifest(report_path)
    manifest["reproducibility_key"] = compute_reproducibility_key(manifest)
    return manifest


class RunManifestTests(unittest.TestCase):
    def test_packaged_schema_is_available(self) -> None:
        schema = load_run_manifest_schema()
        self.assertEqual(schema["properties"]["schema_version"]["const"], RUN_MANIFEST_SCHEMA_VERSION)
        self.assertEqual(
            schema["x-wind3dgs-normative-validator"],
            "wind3dgs.contracts.validate_run_manifest",
        )

    def test_round_trip_and_output_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.json"
            write_canonical_json(report, {"status": "pass"})
            manifest = finalized_manifest(report)
            validate_run_manifest(manifest)
            manifest_path = root / "run_manifest.json"
            write_canonical_json(manifest_path, manifest)
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded, manifest)
            verify_manifest_outputs(loaded, root)

    def test_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_text("{}\n", encoding="utf-8")
            manifest = finalized_manifest(report)
            manifest["outputs"][0]["path"] = "../report.json"
            with self.assertRaisesRegex(ContractError, "contained relative"):
                validate_run_manifest(manifest)

    def test_td01_cannot_hide_missing_object_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_text("{}\n", encoding="utf-8")
            manifest = finalized_manifest(report)
            manifest["milestone"] = "TD01"
            with self.assertRaisesRegex(ContractError, "only for TD00"):
                validate_run_manifest(manifest)

    def test_td01_requires_a_real_object_package_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_text("{}\n", encoding="utf-8")
            manifest = finalized_manifest(report)
            manifest["milestone"] = "TD01"
            manifest["dataset_id"] = "fixture-dataset"
            manifest["object_package_id"] = "fixture-package"
            manifest["object_package_sha256"] = "garbage"
            with self.assertRaisesRegex(ContractError, "SHA-256"):
                validate_run_manifest(manifest)

    def test_created_at_requires_time_and_timezone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_text("{}\n", encoding="utf-8")
            manifest = finalized_manifest(report)
            manifest["created_at"] = "2026-08-13"
            with self.assertRaisesRegex(ContractError, "RFC-3339"):
                validate_run_manifest(manifest)

    def test_boolean_output_size_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_text("{}\n", encoding="utf-8")
            manifest = finalized_manifest(report)
            manifest["outputs"][0]["size_bytes"] = True
            with self.assertRaisesRegex(ContractError, "nonnegative integer"):
                validate_run_manifest(manifest)

    def test_duplicate_config_argument_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_text("{}\n", encoding="utf-8")
            manifest = finalized_manifest(report)
            manifest["command"].extend(("--config", manifest["config_path"]))
            with self.assertRaisesRegex(ContractError, "exactly one --config"):
                validate_run_manifest(manifest)

    def test_symlink_output_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.json"
            target = root / "target.json"
            target.write_text("{}\n", encoding="utf-8")
            report.symlink_to(target.name)
            manifest = finalized_manifest(target)
            manifest["outputs"][0]["path"] = report.name
            manifest["outputs"][0]["sha256"] = sha256_file(target)
            manifest["outputs"][0]["size_bytes"] = target.stat().st_size
            validate_run_manifest(manifest)
            with self.assertRaisesRegex(ContractError, "symlink"):
                verify_manifest_outputs(manifest, root)

    def test_reproducibility_key_detects_command_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_text("{}\n", encoding="utf-8")
            manifest = finalized_manifest(report)
            manifest["command"].append("--changed")
            with self.assertRaisesRegex(ContractError, "reproducibility_key"):
                validate_run_manifest(manifest)

    def test_config_path_must_match_command_and_file_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            config = workspace / "code/configs/td00_contracts_smoke.json"
            config.parent.mkdir(parents=True)
            config.write_text("{}\n", encoding="utf-8")
            report = workspace / "report.json"
            report.write_text("{}\n", encoding="utf-8")
            manifest = finalized_manifest(report)
            manifest["config_sha256"] = sha256_file(config)
            manifest["reproducibility_key"] = compute_reproducibility_key(manifest)
            validate_run_manifest(manifest)
            verify_manifest_config(manifest, workspace)
            manifest["command"][-1] = "code/configs/other.json"
            manifest["reproducibility_key"] = compute_reproducibility_key(manifest)
            with self.assertRaisesRegex(ContractError, "must match config_path"):
                validate_run_manifest(manifest)

    def test_config_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            target = workspace / "target.json"
            target.write_text("{}\n", encoding="utf-8")
            config = workspace / "code/configs/td00_contracts_smoke.json"
            config.parent.mkdir(parents=True)
            config.symlink_to(target)
            report = workspace / "report.json"
            report.write_text("{}\n", encoding="utf-8")
            manifest = finalized_manifest(report)
            manifest["config_sha256"] = sha256_file(target)
            manifest["reproducibility_key"] = compute_reproducibility_key(manifest)
            validate_run_manifest(manifest)
            with self.assertRaisesRegex(ContractError, "symlink"):
                verify_manifest_config(manifest, workspace)


if __name__ == "__main__":
    unittest.main()
