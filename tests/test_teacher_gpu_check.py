from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from wind3dgs.evaluation import teacher_gpu_check as check


class TeacherGPUCheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_cuda_is_required_and_memory_roundtrip_happens(self):
        with patch.dict(sys.modules, {"warp": SimpleNamespace(init=lambda: None, get_cuda_devices=lambda: [],
                                                               get_cuda_driver_version=lambda: None)}):
            with self.assertRaisesRegex(RuntimeError, "CPU fallback"):
                check._select_cuda("cuda:0", self.root)
        self.assertEqual(json.loads((self.root / "cuda.json").read_text())["cuda_devices"], [])
        class Device:
            name, arch, is_cuda = "test GPU", 90, True
            def __str__(self): return "cuda:0"
        calls = []
        fake = SimpleNamespace(init=lambda: None, get_cuda_devices=lambda: [Device()],
                               get_cuda_driver_version=lambda: (12, 9), get_device=lambda name: Device(), float32="float32",
                               zeros=lambda *a, **kw: SimpleNamespace(numpy=lambda: SimpleNamespace(tolist=lambda: [0.])),
                               synchronize_device=lambda device: calls.append(str(device)))
        with patch.dict(sys.modules, {"warp": fake}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(check._select_cuda("cuda:0", self.root), "cuda:0")
        self.assertEqual(calls, ["cuda:0"])

    def test_preflight_failure_keeps_all_stages_not_started(self):
        with patch.object(check, "_select_cuda", side_effect=RuntimeError("no GPU")), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(check._worker(self.root, "cuda:0"), 1)
        result = json.loads((self.root / "checks.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["active_stage"], "cuda_preflight")
        self.assertTrue(all(s["status"] == "not_started" for s in result["stages"].values()))

    def supervise_fixture(self, output, body):
        command = [sys.executable, "-u", "-c", body, str(output), json.dumps(check.STAGES)]
        with patch.object(check, "_worker_command", return_value=command), patch.object(check, "_nvidia_smi", return_value={}), \
                contextlib.redirect_stdout(io.StringIO()):
            return check._supervise(output, "cuda:0")

    def test_supervisor_captures_both_streams_redacts_paths_and_retains_failure_exit(self):
        output = self.root / "failed"
        script = "import sys; print(sys.argv[1]); print('native failure',file=sys.stderr); raise SystemExit(7)"
        self.assertEqual(self.supervise_fixture(output, script), 1)
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["child_exit_code"], 7)
        self.assertEqual(summary["status"], "failed")
        log = (output / "run.log").read_text()
        self.assertIn("native failure", log)
        self.assertIn("<run>", log)
        self.assertNotIn(str(output), log)
        with self.assertRaises(FileExistsError):
            check._supervise(output, "cuda:0")

    def test_zero_exit_without_all_stages_is_failure_and_full_inventory_passes(self):
        self.assertEqual(self.supervise_fixture(self.root / "empty", "pass"), 1)
        script = '''
import json, pathlib, sys
output=pathlib.Path(sys.argv[1]); names=json.loads(sys.argv[2])
(output/'checks.json').write_text(json.dumps({'status':'passed','active_stage':None,'stages':{n:{'status':'passed'} for n in names}}))
'''
        output = self.root / "passed"
        self.assertEqual(self.supervise_fixture(output, script), 0)
        self.assertEqual(len(json.loads((output / "summary.json").read_text())["completed_stages"]), 12)

    def test_device_alias_and_existing_output_rejected_and_launcher_help(self):
        for argv in (["--device", "cpu"], ["--output", str(self.root)]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                check.main(argv)
            self.assertEqual(raised.exception.code, 2)
        launcher = check.CODE_ROOT / "scripts/check_teacher_gpu.sh"
        result = subprocess.run(["bash", str(launcher), "--help"], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--device", result.stdout)


class TeacherGPUCheckPipelineTests(unittest.TestCase):
    def test_actual_pipeline_on_cpu_fixture_without_claiming_gpu_validation(self):
        try:
            import newton  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("optional Newton dependency is not installed")
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            checks = {"status": "cpu_test_only", "stages": {name: {"status": "not_started"} for name in check.STAGES}}
            check._execute_checks(root, "cpu", checks)
            self.assertEqual(len(checks["stages"]), 12)
            self.assertTrue(all(s["status"] == "passed" for s in checks["stages"].values()))
            self.assertEqual(checks["status"], "cpu_test_only")
            for name in check.STAGES[:7]:
                self.assertEqual(checks["stages"][name]["result"]["device"], "cpu")
            self.assertEqual(checks["stages"]["compare_spatial"]["result"]["convergence_status"], "not_assessed")


if __name__ == "__main__":
    unittest.main()
