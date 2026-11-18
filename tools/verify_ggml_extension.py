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
    if not callable(getattr(module, "collect_performance_stats", None)) or not callable(
        getattr(module, "reset_performance_stats", None)
    ):
        raise RuntimeError("Native extension does not expose the optional profiling API; rebuild the extension")
    if not callable(getattr(module, "linear_glu_mmq", None)) or not hasattr(module, "GGMLGluOp"):
        raise RuntimeError("Native extension does not expose the fused gate/up MMQ API; rebuild the extension")

    info = module.build_info()
    if info.get("fusion_api_version") != 1:
        raise RuntimeError("Native extension fused gate/up API version does not match this source")
    if tuple(info.get("fusion_weight_types", ())) != ("Q4_K", "Q5_K", "Q6_K", "Q8_0", "IQ4_XS"):
        raise RuntimeError("Native extension fused weight types do not match this source")
    if tuple(info.get("fusion_glu_ops", ())) != ("GEGLU", "SWIGLU", "SWIGLU_OAI", "SWIGLU_CLAMP"):
        raise RuntimeError("Native extension fused GLU operations do not match this source")
    if any(not hasattr(module.GGMLGluOp, name) for name in info["fusion_glu_ops"]):
        raise RuntimeError("Native extension is missing a fused GLU enum value")
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
