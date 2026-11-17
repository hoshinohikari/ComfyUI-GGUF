"""Compute a stable ID for the ggml native extension source contract."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    Path("native/CMakeLists.txt"),
    Path("native/ggml_torch.cpp"),
    Path("native/ggml/comfyui_gguf_mmq_external.inc"),
    Path("tools/patch_ggml_source.py"),
)


def compute_build_id() -> str:
    digest = hashlib.sha256()
    for relative in FILES:
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update((ROOT / relative).read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


if __name__ == "__main__":
    print(compute_build_id())
