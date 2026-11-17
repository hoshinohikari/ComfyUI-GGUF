"""Import-check the built extension without invoking a CUDA kernel."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extension_directory", type=Path)
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--torch-build", required=True)
    parser.add_argument("--cuda-archs", required=True)
    args = parser.parse_args()

    binaries = list(args.extension_directory.glob("_comfyui_gguf_ggml*.pyd"))
    if len(binaries) != 1:
        raise RuntimeError(f"Expected one native extension, found {len(binaries)}")
    spec = importlib.util.spec_from_file_location("_comfyui_gguf_ggml", binaries[0])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot create an import spec for {binaries[0]}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    info = module.build_info()
    expected_archs = sorted(int(a) for a in args.cuda_archs.replace(",", ";").split(";"))
    if sorted(info.get("cuda_archs", [])) != expected_archs:
        raise RuntimeError("Native extension architecture list does not match this build")
    if info.get("build_id") != args.build_id:
        raise RuntimeError("Native extension build ID does not match the current source")
    if info.get("torch_build") != args.torch_build or torch.__version__ != args.torch_build:
        raise RuntimeError(
            f"Torch build mismatch: extension={info.get('torch_build')!r}, "
            f"runtime={torch.__version__!r}, expected={args.torch_build!r}"
        )
    print(
        "Import check passed without running a CUDA kernel: "
        f"ggml={info['ggml_commit']} CUDA={info['cuda_runtime']} "
        f"SMs={info['cuda_archs']} Torch={info['torch_build']}"
    )


if __name__ == "__main__":
    main()
