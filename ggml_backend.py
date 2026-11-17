"""Load and validate the optional pinned ggml CUDA MMQ extension."""

from __future__ import annotations

import importlib.util
import hashlib
import logging
import os
from pathlib import Path
import sys
import threading
from typing import Any

import gguf
import torch


EXPECTED_GGML_COMMIT = "d9fe7bff3d2bf4d077e25c654486d02c321734aa"
SUPPORTED_CUDA_ARCHS = frozenset({75, 86, 89, 120})
WORKSPACE_LIMIT_BYTES = 64 * 1024 * 1024
GGML_MMQ_WEIGHT_TYPES = (
    "Q1_0", "Q2_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K",
    "IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_XS", "IQ4_NL",
)
SUPPORTED_WEIGHT_TYPES = frozenset(
    name for name in GGML_MMQ_WEIGHT_TYPES if hasattr(gguf.GGMLQuantizationType, name)
)
_module: Any | None = None
_module_lock = threading.Lock()
_device_lock = threading.Lock()
_logged: set[tuple[Any, ...]] = set()
_checked_devices: set[int] = set()
_dll_directory_handles: list[Any] = []


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
    module = _load_extension()
    try:
        return module.linear_mmq(
            quantized_weight,
            input_2d,
            int(output_features),
            int(type_id),
            WORKSPACE_LIMIT_BYTES,
        )
    except ValueError as exc:
        message = str(exc)
        if message.startswith("GGML_MMQ_UNSUPPORTED:"):
            raise GGMLMMQUnsupported(message.partition(":")[2].strip()) from exc
        raise


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
