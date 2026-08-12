from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import wind3dgs

from wind3dgs.contracts import ContractError
from wind3dgs.runtime.td00_smoke import (
    artifact_run_root_is_ignored,
    require_unchanged_repository_snapshots,
    run_smoke,
    validate_executing_package,
    validate_td00_manifest_semantics,
    validate_td00_config,
    verify_td00_evidence_directory,
)


def initialize_repository(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Wind3DGS Test",
            "-c",
            "user.email=wind3dgs-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        cwd=path,
        check=True,
    )


class Td00SmokeTests(unittest.TestCase):
    def test_clean_four_repository_smoke_and_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            code = workspace / "code"
            ideas = workspace / "ideas"
            experiments = workspace / "experiments"
            for path in (code, ideas, experiments):
                path.mkdir()

            (workspace / ".gitignore").write_text("/code/\n/ideas/\n/experiments/\n", encoding="utf-8")
            manifests = workspace / "manifests"
            manifests.mkdir()
            (manifests / "artifact_policy.json").write_text(
                json.dumps({"physical_layout": {"shared_heavy_root": "experiments/artifacts/"}}),
                encoding="utf-8",
            )
            (workspace / "README.md").write_text("fixture\n", encoding="utf-8")

            package_root = code / "wind3dgs"
            package_root.mkdir()
            (package_root / "__init__.py").write_text("\n", encoding="utf-8")
            mainline = [
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
            ]
            for package in mainline:
                package_path = package_root / package
                package_path.mkdir()
                (package_path / ".gitkeep").write_text("", encoding="utf-8")
            (code / "README.md").write_text(
                "<!-- td00:legacy-results-not-evidence -->\n", encoding="utf-8"
            )
            config_directory = code / "configs"
            config_directory.mkdir()
            config = {
                "schema_version": "wind3dgs.td00_smoke_config.v1",
                "milestone": "TD00",
                "experiment_id": "TD00_contracts",
                "seed": 11,
                "device": "cpu",
                "require_clean_sources": True,
                "inherit_legacy_completion": False,
                "artifact_run_root": "experiments/artifacts/runs",
                "source_repositories": [
                    {"id": "project", "relative_path": "."},
                    {"id": "code", "relative_path": "code"},
                    {"id": "ideas", "relative_path": "ideas"},
                    {"id": "experiments", "relative_path": "experiments"},
                ],
                "dataset_id": "not_applicable:td00_governance_smoke",
                "dataset_sha256_or_manifest_version": "manifest:manifests/datasets.json@1.0.0",
                "object_package_id": "not_applicable:td00_governance_smoke",
                "object_package_sha256": "not_applicable:td00_governance_smoke",
                "models": [],
                "expected_stage_order": [1, 2, 3, 4, 5, 6, 7, 10, 9, 8, 11, 12, 13, 14],
                "mainline_packages": mainline,
                "publication_max_file_bytes": 1048576,
            }
            (config_directory / "td00_contracts_smoke.json").write_text(
                json.dumps(config, sort_keys=True), encoding="utf-8"
            )

            (ideas / "README.md").write_text("current ideas\n", encoding="utf-8")
            (experiments / ".gitignore").write_text("artifacts/\n", encoding="utf-8")
            (experiments / "README.md").write_text(
                "<!-- td00:legacy-status-not-inherited -->\n", encoding="utf-8"
            )
            reference = experiments / "TD00_contracts/reports/reference_smoke"
            reference.mkdir(parents=True)
            (reference / "README.md").write_text("reference\n", encoding="utf-8")
            (experiments / "EXPERIMENT_TEMPLATE.md").write_text("template\n", encoding="utf-8")
            td00_root = experiments / "TD00_contracts"
            (td00_root / "README.md").write_text("td00\n", encoding="utf-8")
            (td00_root / "reports/reuse_candidate_audit.md").write_text("audit\n", encoding="utf-8")

            for repository in (code, ideas, experiments, workspace):
                initialize_repository(repository)

            with mock.patch.object(wind3dgs, "__file__", str(package_root / "__init__.py")):
                run_directory, manifest = run_smoke(
                    workspace=workspace,
                    config_relative="code/configs/td00_contracts_smoke.json",
                    publish_relative="experiments/TD00_contracts/reports/reference_smoke",
                )
            self.assertTrue((run_directory / "run_manifest.json").is_file())
            self.assertTrue((run_directory / "td00_smoke_report.json").is_file())
            self.assertTrue((reference / "run_manifest.json").is_file())
            self.assertTrue((reference / "td00_smoke_report.json").is_file())
            self.assertEqual(manifest["milestone"], "TD00")
            self.assertTrue(all(not record["dirty"] for record in manifest["source_repositories"]))

    def test_dependency_scan_detects_legacy_import_forms(self) -> None:
        from wind3dgs.runtime.td00_smoke import scan_mainline_imports

        with tempfile.TemporaryDirectory() as directory:
            code_root = Path(directory)
            package_root = code_root / "wind3dgs"
            mainline = package_root / "runtime"
            mainline.mkdir(parents=True)
            (package_root / "__init__.py").write_text(
                "from wind3dgs import m03_procedural_wind\n", encoding="utf-8"
            )
            (mainline / "bad.py").write_text(
                "from wind3dgs.m02_mesh_proxy_binding import viewer_gpu\n"
                "import importlib\n"
                "importlib.import_module('wind3dgs.m01_static_3dgs_io')\n"
                "from .. import evaluation\n"
                "from wind3dgs import evaluation\n"
                "importlib.import_module('..evaluation', package=__package__)\n"
                "name = 'wind3dgs.m04_mesh_extraction'\n"
                "importlib.import_module(name)\n",
                encoding="utf-8",
            )
            violations = scan_mainline_imports(code_root, ["runtime"])
            self.assertGreaterEqual(len(violations), 7)
            self.assertTrue(any("m02_mesh_proxy_binding" in value for value in violations))
            self.assertTrue(any("m03_procedural_wind" in value for value in violations))
            self.assertTrue(any("dynamic" in value and "m01_static_3dgs_io" in value for value in violations))
            self.assertTrue(any("evaluation" in value for value in violations))
            self.assertTrue(any("relative dynamic import" in value for value in violations))
            self.assertTrue(any("non-literal dynamic import" in value for value in violations))

    def test_config_cannot_raise_publication_limit(self) -> None:
        config = json.loads(
            (Path(__file__).resolve().parents[1] / "configs/td00_contracts_smoke.json").read_text(
                encoding="utf-8"
            )
        )
        config["publication_max_file_bytes"] = 10**12
        with self.assertRaisesRegex(ContractError, "1048576"):
            validate_td00_config(config)

    def test_repository_snapshot_change_is_rejected(self) -> None:
        expected = [{"id": "code", "relative_path": "code", "commit": "a" * 40, "dirty": False}]
        actual = [{"id": "code", "relative_path": "code", "commit": "a" * 40, "dirty": True}]
        with self.assertRaisesRegex(ContractError, "변경"):
            require_unchanged_repository_snapshots(expected, actual)

    def test_config_cannot_redirect_split_repository_ids(self) -> None:
        config = json.loads(
            (Path(__file__).resolve().parents[1] / "configs/td00_contracts_smoke.json").read_text(
                encoding="utf-8"
            )
        )
        config["source_repositories"] = [
            {"id": repository_id, "relative_path": "."}
            for repository_id in ("project", "code", "ideas", "experiments")
        ]
        with self.assertRaisesRegex(ContractError, "split-repository"):
            validate_td00_config(config)

    def test_config_cannot_shrink_dependency_scan(self) -> None:
        config = json.loads(
            (Path(__file__).resolve().parents[1] / "configs/td00_contracts_smoke.json").read_text(
                encoding="utf-8"
            )
        )
        config["mainline_packages"] = ["contracts"]
        with self.assertRaisesRegex(ContractError, "complete TD00"):
            validate_td00_config(config)

    def test_artifact_run_root_must_be_git_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            experiments = workspace / "experiments"
            experiments.mkdir()
            (experiments / "README.md").write_text("fixture\n", encoding="utf-8")
            initialize_repository(experiments)
            self.assertFalse(
                artifact_run_root_is_ignored(workspace, "experiments/artifacts/runs")
            )

    def test_executing_package_must_match_recorded_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "code/wind3dgs").mkdir(parents=True)
            with self.assertRaisesRegex(ContractError, "package"):
                validate_executing_package(workspace)

    def test_evidence_directory_rejects_failed_report(self) -> None:
        from wind3dgs.contracts import (
            RUN_MANIFEST_SCHEMA_VERSION,
            compute_reproducibility_key,
            sha256_file,
            write_canonical_json,
        )
        from wind3dgs.runtime.td00_smoke import TD00_NOT_EVALUATED, TD00_REQUIRED_CHECKS, TD00_SCOPE

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "td00_smoke_report.json"
            report = {
                "schema_version": "wind3dgs.td00_smoke_report.v1",
                "run_id": "td00-run",
                "milestone": "TD00",
                "status": "pass",
                "scope": TD00_SCOPE,
                "checks": {
                    key: {"status": "pass", "message": "fixture 통과", "evidence": key}
                    for key in TD00_REQUIRED_CHECKS
                },
                "not_evaluated": TD00_NOT_EVALUATED,
            }
            write_canonical_json(report_path, report)
            manifest = {
                "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
                "run_id": "td00-run",
                "milestone": "TD00",
                "created_at": "2026-08-13T00:00:00Z",
                "source_repositories": [
                    {"id": key, "relative_path": value, "commit": "a" * 40, "dirty": False}
                    for key, value in {
                        "project": ".",
                        "code": "code",
                        "ideas": "ideas",
                        "experiments": "experiments",
                    }.items()
                ],
                "command": [
                    ".venv/bin/python",
                    "-m",
                    "wind3dgs.runtime.td00_smoke",
                    "--config",
                    "code/configs/td00_contracts_smoke.json",
                ],
                "working_directory": ".",
                "environment": {"PYTHONPATH": "code"},
                "config_path": "code/configs/td00_contracts_smoke.json",
                "config_sha256": "b" * 64,
                "seed": 1,
                "device": {
                    "requested": "cpu",
                    "resolved": "cpu",
                    "class": "cpu-contract-smoke",
                },
                "dataset_id": "not_applicable:td00_governance_smoke",
                "dataset_sha256_or_manifest_version": "manifest:datasets@1",
                "object_package_id": "not_applicable:td00_governance_smoke",
                "object_package_sha256": "not_applicable:td00_governance_smoke",
                "models": [],
                "outputs": [
                    {
                        "id": "td00_smoke_report",
                        "path": report_path.name,
                        "sha256": sha256_file(report_path),
                        "size_bytes": report_path.stat().st_size,
                    }
                ],
                "software": {"python": "3.12", "numpy": "2", "platform": "test"},
                "reproducibility_key": "0" * 64,
            }
            manifest["reproducibility_key"] = compute_reproducibility_key(manifest)
            manifest_path = root / "run_manifest.json"
            completion_path = root / "reference_complete.json"
            write_canonical_json(manifest_path, manifest)
            write_canonical_json(
                completion_path,
                {
                    "schema_version": "wind3dgs.td00_reference_complete.v1",
                    "run_id": manifest["run_id"],
                    "manifest_sha256": sha256_file(manifest_path),
                    "report_sha256": sha256_file(report_path),
                },
            )
            verify_td00_evidence_directory(root)

            report["status"] = "fail"
            write_canonical_json(report_path, report)
            manifest["outputs"][0]["sha256"] = sha256_file(report_path)
            manifest["outputs"][0]["size_bytes"] = report_path.stat().st_size
            write_canonical_json(manifest_path, manifest)
            write_canonical_json(
                completion_path,
                {
                    "schema_version": "wind3dgs.td00_reference_complete.v1",
                    "run_id": manifest["run_id"],
                    "manifest_sha256": sha256_file(manifest_path),
                    "report_sha256": sha256_file(report_path),
                },
            )
            with self.assertRaisesRegex(ContractError, "status"):
                verify_td00_evidence_directory(root)

    def test_evidence_directory_rechecks_privacy_policy(self) -> None:
        from wind3dgs.contracts import (
            RUN_MANIFEST_SCHEMA_VERSION,
            compute_reproducibility_key,
            sha256_file,
            write_canonical_json,
        )
        from wind3dgs.runtime.td00_smoke import TD00_NOT_EVALUATED, TD00_REQUIRED_CHECKS, TD00_SCOPE

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "td00_smoke_report.json"
            checks = {
                key: {"status": "pass", "message": "fixture 통과", "evidence": key}
                for key in TD00_REQUIRED_CHECKS
            }
            checks["no_absolute_path_or_secret"]["evidence"] = {"api_key": "should-not-appear"}
            report = {
                "schema_version": "wind3dgs.td00_smoke_report.v1",
                "run_id": "td00-private-run",
                "milestone": "TD00",
                "status": "pass",
                "scope": TD00_SCOPE,
                "checks": checks,
                "not_evaluated": TD00_NOT_EVALUATED,
            }
            write_canonical_json(report_path, report)
            manifest = {
                "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
                "run_id": "td00-private-run",
                "milestone": "TD00",
                "created_at": "2026-08-13T00:00:00Z",
                "source_repositories": [
                    {"id": key, "relative_path": value, "commit": "a" * 40, "dirty": False}
                    for key, value in {
                        "project": ".",
                        "code": "code",
                        "ideas": "ideas",
                        "experiments": "experiments",
                    }.items()
                ],
                "command": [
                    ".venv/bin/python",
                    "-m",
                    "wind3dgs.runtime.td00_smoke",
                    "--config",
                    "code/configs/td00_contracts_smoke.json",
                ],
                "working_directory": ".",
                "environment": {"PYTHONPATH": "code"},
                "config_path": "code/configs/td00_contracts_smoke.json",
                "config_sha256": "b" * 64,
                "seed": 1,
                "device": {"requested": "cpu", "resolved": "cpu", "class": "cpu-contract-smoke"},
                "dataset_id": "not_applicable:td00_governance_smoke",
                "dataset_sha256_or_manifest_version": "manifest:datasets@1",
                "object_package_id": "not_applicable:td00_governance_smoke",
                "object_package_sha256": "not_applicable:td00_governance_smoke",
                "models": [],
                "outputs": [{
                    "id": "td00_smoke_report",
                    "path": report_path.name,
                    "sha256": sha256_file(report_path),
                    "size_bytes": report_path.stat().st_size,
                }],
                "software": {"python": "3.12", "numpy": "2", "platform": "test"},
                "reproducibility_key": "0" * 64,
            }
            manifest["reproducibility_key"] = compute_reproducibility_key(manifest)
            manifest_path = root / "run_manifest.json"
            write_canonical_json(manifest_path, manifest)
            write_canonical_json(
                root / "reference_complete.json",
                {
                    "schema_version": "wind3dgs.td00_reference_complete.v1",
                    "run_id": manifest["run_id"],
                    "manifest_sha256": sha256_file(manifest_path),
                    "report_sha256": sha256_file(report_path),
                },
            )
            with self.assertRaisesRegex(ContractError, "secret-like"):
                verify_td00_evidence_directory(root)

    def test_td00_manifest_semantics_reject_dirty_source(self) -> None:
        manifest = {
            "milestone": "TD00",
            "source_repositories": [
                {"id": key, "relative_path": value, "dirty": key == "code"}
                for key, value in {
                    "project": ".",
                    "code": "code",
                    "ideas": "ideas",
                    "experiments": "experiments",
                }.items()
            ],
            "device": {"requested": "cpu", "resolved": "cpu", "class": "cpu-contract-smoke"},
            "working_directory": ".",
            "environment": {"PYTHONPATH": "code"},
            "command": [".venv/bin/python", "-m", "wind3dgs.runtime.td00_smoke"],
            "config_path": "code/configs/td00_contracts_smoke.json",
        }
        with self.assertRaisesRegex(ContractError, "dirty"):
            validate_td00_manifest_semantics(manifest)


if __name__ == "__main__":
    unittest.main()
