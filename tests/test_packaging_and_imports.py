from __future__ import annotations

import ast
import importlib
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


class PackagingAndImportTests(unittest.TestCase):
    def test_td_core_imports_without_renderer_or_torch(self) -> None:
        for package in (
            "wind3dgs.contracts",
            "wind3dgs.io",
            "wind3dgs.teacher",
            "wind3dgs.topology",
            "wind3dgs.aero",
            "wind3dgs.reduced",
            "wind3dgs.local",
            "wind3dgs.learning",
            "wind3dgs.runtime",
            "wind3dgs.transport",
            "wind3dgs.evaluation",
        ):
            with self.subTest(package=package):
                importlib.import_module(package)

    def test_td_core_import_does_not_load_optional_gpu_modules(self) -> None:
        code_root = Path(__file__).resolve().parents[1]
        script = """
import builtins
import importlib

blocked = {"torch", "gsplat", "moderngl", "glfw", "glcontext", "PIL", "scipy", "yaml", "OpenGL", "newton", "warp"}
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".", 1)[0] in blocked:
        raise ImportError(f"optional module blocked during TD core import: {name}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
for package in (
    "wind3dgs.contracts",
    "wind3dgs.io",
    "wind3dgs.teacher",
    "wind3dgs.topology",
    "wind3dgs.aero",
    "wind3dgs.reduced",
    "wind3dgs.local",
    "wind3dgs.learning",
    "wind3dgs.runtime",
    "wind3dgs.transport",
    "wind3dgs.evaluation",
):
    importlib.import_module(package)
loaded_optional = sorted(name for name in sys.modules if name.split(".", 1)[0] in blocked)
if loaded_optional:
    raise RuntimeError(f"TD core loaded optional GPU modules: {loaded_optional}")
"""
        result = subprocess.run(
            [sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(code_root)!r});\n{script}"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_built_wheel_contains_contract_resources_and_imports_core(self) -> None:
        code_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            shutil.copytree(
                code_root,
                source,
                ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "outputs", "datasets", "sessions"),
            )
            wheel_root = root / "wheels"
            wheel_root.mkdir()
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-deps",
                    "--no-build-isolation",
                    "--wheel-dir",
                    str(wheel_root),
                    str(source),
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            wheel = next(wheel_root.glob("wind3dgs-*.whl"))
            with zipfile.ZipFile(wheel) as archive:
                names = set(archive.namelist())
            self.assertIn("wind3dgs/py.typed", names)
            self.assertIn("wind3dgs/contracts/schemas/run_manifest.schema.json", names)

            target = root / "installed"
            install = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target), str(wheel)],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(install.returncode, 0, install.stderr)
            probe = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    (
                        f"import sys; sys.path.insert(0, {str(target)!r}); "
                        "from wind3dgs.contracts import load_run_manifest_schema; "
                        "assert load_run_manifest_schema()['x-wind3dgs-normative-validator']; "
                        "import wind3dgs.runtime, wind3dgs.io, wind3dgs.aero, wind3dgs.transport"
                    ),
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(probe.returncode, 0, probe.stderr)

    def test_all_package_sources_parse_as_python_310(self) -> None:
        package_root = Path(__file__).resolve().parents[1] / "wind3dgs"
        for path in sorted(package_root.rglob("*.py")):
            with self.subTest(path=path.relative_to(package_root)):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))


if __name__ == "__main__":
    unittest.main()
