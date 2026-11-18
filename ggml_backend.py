"""Load and validate the optional pinned ggml CUDA MMQ extension."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import threading
from collections import defaultdict
from typing import Any

import gguf
import torch


EXPECTED_GGML_COMMIT = "d9fe7bff3d2bf4d077e25c654486d02c321734aa"
SUPPORTED_CUDA_ARCHS = frozenset({75, 86, 89, 120})
WORKSPACE_BUDGET_ENV = "COMFYUI_GGUF_WORKSPACE_BUDGET_MIB"
_MIB_BYTES = 1024 * 1024
_MAX_WORKSPACE_BUDGET_MIB = ((1 << 63) - 1) // _MIB_BYTES
PERF_STATS_ENABLED = os.environ.get("COMFYUI_GGUF_PERF_STATS", "").strip().lower() in {"1", "true", "yes", "on"}
GGML_MMQ_WEIGHT_TYPES = (
    "Q1_0", "Q2_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K",
    "IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_XS", "IQ4_NL",
)
GGML_MMQ_FUSION_WEIGHT_TYPES = ("Q4_K", "Q5_K", "Q6_K", "Q8_0", "IQ4_XS")
GGML_MMQ_FUSION_GLU_OPS = ("GEGLU", "SWIGLU", "SWIGLU_OAI", "SWIGLU_CLAMP")
SUPPORTED_WEIGHT_TYPES = frozenset(
    name for name in GGML_MMQ_WEIGHT_TYPES if hasattr(gguf.GGMLQuantizationType, name)
)
_module: Any | None = None
_module_lock = threading.Lock()
_device_lock = threading.Lock()
_logged: set[tuple[Any, ...]] = set()
_checked_devices: set[int] = set()
_dll_directory_handles: list[Any] = []
_perf_lock = threading.Lock()
_perf_fallbacks: dict[tuple[Any, ...], int] = defaultdict(int)
_workspace_budget_lock = threading.Lock()
_workspace_budget_cache: int | None = None

if PERF_STATS_ENABLED:
    logging.info(
        "ComfyUI-GGUF MMQ performance statistics enabled; use POST /gguf/perf/reset before a prompt and POST /gguf/perf/report after it"
    )


def record_perf_fallback(qtype: str, input_tensor: torch.Tensor, output_features: int, reason: str) -> None:
    if not PERF_STATS_ENABLED:
        return
    shape = tuple(int(v) for v in input_tensor.shape)
    rows = 1
    for value in shape[:-1]:
        rows *= value
    key = (
        qtype or "unknown",
        rows,
        int(output_features),
        int(input_tensor.shape[-1]) if shape else 0,
        str(input_tensor.dtype),
        str(input_tensor.device),
        reason,
    )
    with _perf_lock:
        _perf_fallbacks[key] += 1


def reset_perf_stats() -> dict[str, Any]:
    if PERF_STATS_ENABLED:
        from . import perf_diagnostics
        perf_diagnostics.reset()
    with _perf_lock:
        _perf_fallbacks.clear()
    module = _module
    if module is not None and PERF_STATS_ENABLED:
        module.reset_performance_stats()
    return {"enabled": PERF_STATS_ENABLED, "reset": True}


def collect_perf_stats() -> dict[str, Any]:
    with _perf_lock:
        fallbacks = [
            {
                "qtype": key[0], "M": key[1], "N": key[2], "K": key[3],
                "input_dtype": key[4], "device": key[5], "reason": key[6], "calls": count,
            }
            for key, count in sorted(_perf_fallbacks.items())
        ]
        _perf_fallbacks.clear()

    native = {
        "linear_shapes": [],
        "timing_samples_per_shape_limit": 3,
        "dropped_event_pairs": 0,
        "gpu_synchronized_devices": 0,
    }
    module = _module
    if module is not None and PERF_STATS_ENABLED:
        native = module.collect_performance_stats()
        for row in native.get("linear_shapes", []):
            qtype_id = int(row.pop("qtype_id"))
            try:
                row["qtype"] = gguf.GGMLQuantizationType(qtype_id).name
            except ValueError:
                row["qtype"] = f"type_{qtype_id}"

    report = {"enabled": PERF_STATS_ENABLED, "linear": native.get("linear_shapes", []), "fallbacks": fallbacks,
              "timing_samples_per_shape_limit": native.get("timing_samples_per_shape_limit", 3),
              "dropped_event_pairs": native.get("dropped_event_pairs", 0),
              "gpu_synchronized_devices": native.get("gpu_synchronized_devices", 0)}
    if PERF_STATS_ENABLED:
        from . import perf_diagnostics
        report["diagnostics"] = perf_diagnostics.collect()
    logging.info("ComfyUI-GGUF MMQ performance report: %s", json.dumps(report, sort_keys=True, separators=(",", ":")))
    return report


def _expected_build_id() -> str:
    root = Path(__file__).resolve().parent
    source_files = (
        "native/CMakeLists.txt",
        "native/ggml_torch.cpp",
        "native/ggml/comfyui_gguf_mmq_external.inc",
        "tools/patch_ggml_source.py",
    )
    digest = hashlib.sha256()
    for relative in source_files:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / relative).read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


class GGMLMMQUnsupported(ValueError):
    """The ggml MMQ backend does not support this operator shape or format."""


def _native_directory() -> Path:
    return Path(__file__).resolve().parent / "native" / "build"


def _cuda_toolkit_bin() -> Path | None:
    candidates: list[Path] = []
    if os.environ.get("CUDA_PATH"):
        candidates.append(Path(os.environ["CUDA_PATH"]) / "bin")
    if os.name == "nt":
        try:
            import winreg

            key_path = r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                for name in ("CUDA_PATH", "CUDA_PATH_V13_5"):
                    try:
                        value, _ = winreg.QueryValueEx(key, name)
                    except OSError:
                        continue
                    candidates.insert(0, Path(value) / "bin")
        except OSError:
            pass
    for candidate in candidates:
        if (candidate / "cudart64_13.dll").exists():
            return candidate
    return next((candidate for candidate in candidates if candidate.is_dir()), None)


def _load_extension() -> Any:
    global _module
    if _module is not None:
        return _module
    with _module_lock:
        if _module is not None:
            return _module

        directory = _native_directory()
        candidates = sorted(directory.glob("_comfyui_gguf_ggml*.pyd"))
        if len(candidates) != 1:
            if not candidates:
                detail = "The native extension has not been built."
            else:
                detail = f"Found multiple native extensions: {', '.join(p.name for p in candidates)}."
            raise RuntimeError(
                "The ggml compute mode is selected, but its native extension is unavailable. "
                f"{detail} Run tools/build_ggml_backend.cmd with the pinned ggml checkout, "
                "the extracted Torch SDK, CUDA 13.5, and the configured Visual Studio 18 toolchain."
            )

        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        for dll_directory in (torch_lib, _cuda_toolkit_bin(), directory):
            if dll_directory is not None and dll_directory.is_dir() and hasattr(os, "add_dll_directory"):
                _dll_directory_handles.append(os.add_dll_directory(str(dll_directory)))

        spec = importlib.util.spec_from_file_location("_comfyui_gguf_ggml", candidates[0])
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load ggml extension at {candidates[0]}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            sys.modules.pop(spec.name, None)
            raise RuntimeError(
                f"Cannot load the ggml native extension at {candidates[0]}. "
                "Check the Torch/CUDA runtime DLLs or rebuild with tools/build_ggml_backend.cmd. "
                f"Loader detail: {exc}"
            ) from exc

        info = module.build_info()
        if info.get("ggml_commit") != EXPECTED_GGML_COMMIT:
            raise RuntimeError(
                f"ggml extension commit mismatch: expected {EXPECTED_GGML_COMMIT}, "
                f"found {info.get('ggml_commit')!r}. Rebuild the native extension."
            )
        if tuple(info.get("weight_types", ())) != GGML_MMQ_WEIGHT_TYPES:
            raise RuntimeError("ggml extension weight-format table does not match this plugin; rebuild the extension")
        if info.get("fusion_api_version") != 1:
            raise RuntimeError("ggml extension does not expose fused gate/up API v1; rebuild the extension")
        if tuple(info.get("fusion_weight_types", ())) != GGML_MMQ_FUSION_WEIGHT_TYPES:
            raise RuntimeError("ggml extension fused weight-format table does not match this plugin; rebuild the extension")
        if tuple(info.get("fusion_glu_ops", ())) != GGML_MMQ_FUSION_GLU_OPS:
            raise RuntimeError("ggml extension fused GLU table does not match this plugin; rebuild the extension")
        expected_build_id = _expected_build_id()
        if info.get("build_id") != expected_build_id:
            raise RuntimeError(
                "The ggml extension was built from a different native source revision. "
                "Run tools/build_ggml_backend.cmd again."
            )
        if info.get("torch_build") != torch.__version__:
            raise RuntimeError(
                f"ggml extension was built for Torch {info.get('torch_build')!r}, "
                f"but this ComfyUI uses {torch.__version__!r}. Rebuild against this installation's SDK."
            )
        runtime_cuda = torch.version.cuda
        extension_cuda = str(info.get("cuda_runtime", ""))
        if not runtime_cuda or runtime_cuda.split(".", 1)[0] != extension_cuda.split(".", 1)[0]:
            raise RuntimeError(
                f"ggml extension CUDA runtime {extension_cuda!r} is incompatible with "
                f"Torch CUDA {runtime_cuda!r}. Rebuild with a matching CUDA 13.x toolkit."
            )
        archs = info.get("cuda_archs", ())
        if not archs or any(type(a) is not int or a not in SUPPORTED_CUDA_ARCHS for a in archs):
            raise RuntimeError("ggml extension has an invalid or outdated architecture list; rebuild the extension")

        logging.info(
            "ComfyUI-GGUF ggml backend ready: ggml=%s CUDA=%s targets=%s Torch=%s",
            info["ggml_commit"], info["cuda_runtime"], archs, info["torch_build"],
        )
        _module = module
        return module


def ensure_backend() -> Any:
    """Load the binary and fail clearly when the selected backend is unavailable."""
    return _load_extension()


def ensure_device(device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError("The ggml compute mode requires a CUDA execution device")
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    if device_index in _checked_devices:
        return
    with _device_lock:
        if device_index in _checked_devices:
            return
        capability = torch.cuda.get_device_capability(device_index)
        arch = capability[0] * 10 + capability[1]
        compiled_archs = _load_extension().build_info()["cuda_archs"]
        if arch not in compiled_archs:
            raise RuntimeError(
                f"The installed ggml extension targets {compiled_archs}; device cuda:{device_index} "
                f"reports SM{arch}. Rebuild the native extension for this GPU architecture."
            )
        _checked_devices.add(device_index)


def linear_mmq(
    quantized_weight: torch.Tensor,
    input_2d: torch.Tensor,
    output_features: int,
    type_id: int,
) -> torch.Tensor:
    workspace_budget_bytes = _get_workspace_budget_bytes()
    module = _load_extension()
    try:
        return module.linear_mmq(
            quantized_weight,
            input_2d,
            int(output_features),
            int(type_id),
            workspace_budget_bytes,
            PERF_STATS_ENABLED,
        )
    except ValueError as exc:
        message = str(exc)
        if message.startswith("GGML_MMQ_UNSUPPORTED:"):
            raise GGMLMMQUnsupported(message.partition(":")[2].strip()) from exc
        raise


def linear_glu_mmq(
    up_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    input_2d: torch.Tensor,
    output_features: int,
    up_type_id: int,
    gate_type_id: int,
    glu_op: str,
    glu_limit: float = 0.0,
) -> torch.Tensor:
    """Run the pinned GGML fused gate/up MMQ path; no model dispatch uses it yet."""
    if int(up_type_id) != int(gate_type_id):
        raise GGMLMMQUnsupported("fused gate/up requires identical quantization types")
    if not all(isinstance(value, torch.Tensor) for value in (up_weight, gate_weight, input_2d)):
        raise GGMLMMQUnsupported("fused gate/up weights and activations must be Torch tensors")
    try:
        qtype = gguf.GGMLQuantizationType(int(up_type_id))
        qtype_name = qtype.name
    except ValueError as exc:
        raise GGMLMMQUnsupported(f"unknown GGUF quantization type id {up_type_id}") from exc
    if qtype_name not in GGML_MMQ_FUSION_WEIGHT_TYPES:
        raise GGMLMMQUnsupported(f"GGML CUDA does not provide fused gate/up for {qtype_name}")
    if not up_weight.is_cuda or not gate_weight.is_cuda or not input_2d.is_cuda:
        raise GGMLMMQUnsupported("fused gate/up requires CUDA weights and activations")
    if up_weight.device != gate_weight.device or up_weight.device != input_2d.device:
        raise GGMLMMQUnsupported("fused gate/up weights and activations must share a device")
    if not up_weight.is_contiguous() or not gate_weight.is_contiguous() or up_weight.shape != gate_weight.shape:
        raise GGMLMMQUnsupported("fused gate/up requires same-shape contiguous packed weights")
    if up_weight.dtype != gate_weight.dtype:
        raise GGMLMMQUnsupported("fused gate/up packed storage dtypes must match")
    if input_2d.ndim != 2 or not input_2d.is_contiguous():
        raise GGMLMMQUnsupported("fused gate/up expects a contiguous two-dimensional activation matrix")
    if input_2d.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise GGMLMMQUnsupported(f"unsupported fused activation dtype {input_2d.dtype}")
    if output_features <= 0 or input_2d.shape[1] <= 0:
        raise GGMLMMQUnsupported("fused gate/up matrix dimensions must be positive")
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    if input_2d.shape[1] % block_size:
        raise GGMLMMQUnsupported(f"fused gate/up K dimension must align to GGML block size {block_size}")
    expected_weight_bytes = (input_2d.shape[1] // block_size) * type_size * int(output_features)
    if any(weight.numel() * weight.element_size() != expected_weight_bytes for weight in (up_weight, gate_weight)):
        raise GGMLMMQUnsupported("packed gate/up byte lengths do not match the declared quantized matrix shape")

    glu_name = str(glu_op).strip().upper().removeprefix("GGML_GLU_OP_")
    if glu_name not in GGML_MMQ_FUSION_GLU_OPS:
        raise GGMLMMQUnsupported(f"unsupported fused GLU operation {glu_name or '<empty>'}")
    glu_limit = float(glu_limit)
    if glu_name == "SWIGLU_CLAMP":
        if not math.isfinite(glu_limit) or glu_limit <= 0.0:
            raise GGMLMMQUnsupported("SWIGLU_CLAMP requires a finite positive glu_limit")
    elif glu_limit != 0.0:
        raise GGMLMMQUnsupported("glu_limit is supported only by SWIGLU_CLAMP; SWIGLU_OAI uses GGML defaults")

    module = _load_extension()
    glu_enum = getattr(module.GGMLGluOp, glu_name, None)
    if glu_enum is None:
        raise RuntimeError(f"native GGML extension is missing GLU operation {glu_name}")
    workspace_budget_bytes = _get_workspace_budget_bytes()
    try:
        output = module.linear_glu_mmq(
            up_weight,
            gate_weight,
            input_2d,
            int(output_features),
            int(up_type_id),
            int(gate_type_id),
            glu_enum,
            glu_limit,
            workspace_budget_bytes,
            PERF_STATS_ENABLED,
        )
    except ValueError as exc:
        message = str(exc)
        if message.startswith("GGML_MMQ_UNSUPPORTED:"):
            raise GGMLMMQUnsupported(message.partition(":")[2].strip()) from exc
        raise
    info_once(
        (qtype_name, str(input_2d.device), str(input_2d.dtype), "fusion", glu_name),
        f"ggml MMQ fused gate/up {qtype_name}/{glu_name}, Q8_1 activations, input={input_2d.dtype}, device={input_2d.device}",
    )
    return output


def _get_workspace_budget_bytes() -> int:
    """Return an explicit MiB budget in bytes, or zero for full-matrix mode."""
    global _workspace_budget_cache
    if _workspace_budget_cache is None:
        with _workspace_budget_lock:
            if _workspace_budget_cache is None:
                raw = os.environ.get(WORKSPACE_BUDGET_ENV)
                if raw is None or not raw.strip():
                    _workspace_budget_cache = 0
                else:
                    value = raw.strip()
                    if not value.isascii() or not value.isdecimal():
                        raise ValueError(
                            f"{WORKSPACE_BUDGET_ENV} must be a positive integer in MiB; "
                            "leave it unset for full-matrix mode"
                        )
                    budget_mib = int(value)
                    if budget_mib <= 0 or budget_mib > _MAX_WORKSPACE_BUDGET_MIB:
                        raise ValueError(
                            f"{WORKSPACE_BUDGET_ENV} must be between 1 and {_MAX_WORKSPACE_BUDGET_MIB} MiB"
                        )
                    _workspace_budget_cache = budget_mib * _MIB_BYTES

    budget_bytes = _workspace_budget_cache
    assert budget_bytes is not None
    if budget_bytes == 0:
        info_once(
            ("workspace_budget", 0),
            "ComfyUI-GGUF MMQ workspace policy: full-matrix execution; no artificial budget. "
            f"Set {WORKSPACE_BUDGET_ENV} to a positive MiB value to enable explicit chunking.",
        )
    else:
        info_once(
            ("workspace_budget", budget_bytes),
            f"ComfyUI-GGUF MMQ workspace budget: {budget_bytes // _MIB_BYTES} MiB from {WORKSPACE_BUDGET_ENV}; "
            "this covers estimated bridge temporary buffers and GGML workspace, not total VRAM use",
        )
    return budget_bytes


def info_once(key: tuple[Any, ...], message: str) -> None:
    if key in _logged:
        return
    with _module_lock:
        if key not in _logged:
            _logged.add(key)
            logging.info(message)


def warn_once(key: tuple[Any, ...], message: str) -> None:
    if key in _logged:
        return
    with _module_lock:
        if key not in _logged:
            _logged.add(key)
            logging.warning("ComfyUI-GGUF ggml MMQ fallback: %s", message)
