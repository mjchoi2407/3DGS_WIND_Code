#!/usr/bin/env python3
"""Print a focused gsplat/CUDA environment check without triggering gsplat JIT build."""

from __future__ import annotations

import os
import shutil
import subprocess

import torch
from torch.utils.cpp_extension import CUDA_HOME


def command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        return "not found"
    return result.stdout.strip() or f"exit code {result.returncode}"


def main() -> None:
    print("## PyTorch")
    print(f"torch.__version__: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"CUDA_HOME: {CUDA_HOME}")
    print(f"env CUDA_HOME: {os.environ.get('CUDA_HOME')}")
    print(f"env TORCH_CUDA_ARCH_LIST: {os.environ.get('TORCH_CUDA_ARCH_LIST')}")
    print()

    print("## CUDA Toolkit")
    print(f"nvcc path: {shutil.which('nvcc')}")
    print(command_output(["nvcc", "--version"]))
    print()

    print("## Visible CUDA Devices")
    if not torch.cuda.is_available():
        print("No CUDA device is visible to PyTorch.")
        return

    for index in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(index)
        capability = torch.cuda.get_device_capability(index)
        print(f"device {index}: {name}, compute capability {capability[0]}.{capability[1]}")
        if capability < (7, 0):
            print("  status: gsplat 1.5.3 build is expected to fail because labeled_partition requires CC >= 7.0.")
        else:
            print("  status: compute capability is high enough for labeled_partition.")


if __name__ == "__main__":
    main()
