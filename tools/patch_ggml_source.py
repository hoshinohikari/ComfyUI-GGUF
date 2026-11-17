"""Apply the pinned, narrow ComfyUI borrowed-pointer bridge to a ggml checkout."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


EXPECTED_COMMIT = "d9fe7bff3d2bf4d077e25c654486d02c321734aa"
BRIDGE_INCLUDE = '#include "comfyui_gguf_mmq_external.inc"'


def apply_patch(source: Path, bridge_include: Path, check_only: bool) -> None:
    source = source.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != EXPECTED_COMMIT:
        raise ValueError(f"Expected ggml {EXPECTED_COMMIT}, found {commit}")

    mmq_path = source / "src" / "ggml-cuda" / "mmq.cu"
    text = mmq_path.read_text(encoding="utf-8")
    if BRIDGE_INCLUDE in text:
        if "if (t && t->buffer && ggml_backend_buffer_get_usage(t->buffer)" not in text:
            raise ValueError("Bridge include exists but borrowed-buffer guard is missing")
        if "#include <map>" not in text:
            if check_only:
                raise ValueError("Bridge include exists but the external-stream map header is missing")
            text = text.replace("#include <unordered_map>\n", "#include <unordered_map>\n#include <map>\n", 1)
            mmq_path.write_text(text, encoding="utf-8", newline="\n")
        if not (mmq_path.parent / bridge_include.name).exists() and not check_only:
            raise ValueError("Bridge include is already applied but its .inc file is missing")
        if not check_only:
            (mmq_path.parent / bridge_include.name).write_bytes(bridge_include.read_bytes())
        print("ggml MMQ bridge already applied")
        return

    include_block = "#include <cstdint>\n"
    bridge_includes = (
        "#include <cstdint>\n"
        "#include <mutex>\n"
        "#include <unordered_map>\n"
        "#include <map>\n"
        "#include <memory>\n"
        "#include <string>\n"
        "#include <stdexcept>\n"
        "#include <algorithm>\n"
        "#include <cstdio>\n"
        "#include <cstring>\n"
    )
    buffer_check = "if (t && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {"
    borrowed_buffer_check = "if (t && t->buffer && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {"
    if text.count(include_block) != 1 or text.count(buffer_check) != 1:
        raise ValueError("Pinned ggml source context differs from the reviewed bridge patch")
    text = text.replace(include_block, bridge_includes, 1)
    text = text.replace(buffer_check, borrowed_buffer_check, 1)
    text = text.rstrip() + "\n\n" + BRIDGE_INCLUDE + "\n"

    if not check_only:
        mmq_path.write_text(text, encoding="utf-8", newline="\n")
        (mmq_path.parent / bridge_include.name).write_bytes(bridge_include.read_bytes())
    print("ggml MMQ bridge can be applied" if check_only else "Applied ggml MMQ bridge")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--bridge-include",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "native" / "ggml" / "comfyui_gguf_mmq_external.inc",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    apply_patch(args.source, args.bridge_include, args.check)


if __name__ == "__main__":
    main()
