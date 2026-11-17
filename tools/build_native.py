"""Build the Windows GGML extension using the active ComfyUI Python."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def parse_archs(value: str) -> list[int]:
    try:
        result = sorted(set(int(v) for v in value.replace(",", ";").split(";")))
    except ValueError as exc:
        raise ValueError("Use --cuda-archs 75,86,89,120 (or a subset)") from exc
    if not result or set(result) - {75, 86, 89, 120}:
        raise ValueError("Supported GeForce targets are 75, 86, 89 and 120")
    return result


def executable(name: str, override: str | None) -> str:
    found = shutil.which(override or name)
    if not found:
        raise ValueError(f"Cannot find {override or name}; supply its path or add it to PATH")
    return str(Path(found).resolve())


def torch_sdk(override: Path | None, torch) -> Path:
    root = (override or Path(torch.__file__).resolve().parent).resolve()
    required = ["include/torch/extension.h", "share/cmake/Torch/TorchConfig.cmake"]
    required += [f"lib/{name}.lib" for name in ("torch", "torch_cpu", "torch_python", "torch_cuda", "c10", "c10_cuda")]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"Torch SDK at {root} lacks {', '.join(missing)}; provide --torch-sdk pointing to a matching extracted wheel's torch directory")
    if override and root != Path(torch.__file__).resolve().parent:
        manifest = root.parent / "torch-sdk-manifest.json"
        if not manifest.is_file():
            raise ValueError(f"External SDK requires {manifest} from tools/extract_torch_sdk.py")
        wheel = json.loads(manifest.read_text(encoding="utf-8"))["wheel"]
        expected = f"torch-{torch.__version__}-"
        if not wheel.lower().startswith(expected.lower()):
            raise ValueError(f"SDK wheel {wheel} does not match active Torch {torch.__version__}")
        from packaging.tags import sys_tags
        from packaging.utils import parse_wheel_filename
        _, _, _, tags = parse_wheel_filename(wheel)
        if not tags.intersection(sys_tags()):
            raise ValueError(f"SDK wheel {wheel} is incompatible with this Python/platform")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ggml-repo", type=Path, default=ROOT / "third_party" / "ggml")
    parser.add_argument("--torch-sdk", type=Path)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build" / "ggml-native")
    parser.add_argument("--patched-source", type=Path, default=ROOT / "build" / "ggml-source")
    parser.add_argument("--cuda-root", type=Path, default=os.environ.get("CUDA_PATH"))
    parser.add_argument("--vs-root", type=Path)
    parser.add_argument("--cmake")
    parser.add_argument("--ninja")
    parser.add_argument("--cuda-archs", default="75,86,89,120", help="RTX 20/30/40/50 targets; comma-separated, default: %(default)s")
    args = parser.parse_args()
    if sys.platform != "win32":
        raise ValueError("This build entry point currently supports Windows only")
    import torch
    sdk = torch_sdk(args.torch_sdk, torch)
    if not torch.version.cuda:
        raise ValueError("The active Python does not have a CUDA-enabled Torch")
    if not args.cuda_root:
        raise ValueError("CUDA_PATH is unset; provide --cuda-root")
    archs = parse_archs(args.cuda_archs)
    nvcc = Path(args.cuda_root) / "bin" / "nvcc.exe"
    codes = subprocess.check_output([str(nvcc), "--list-gpu-code"], text=True).split()
    missing = [a for a in archs if f"sm_{a}" not in codes]
    if missing:
        raise ValueError(f"CUDA compiler does not support targets {missing}; select a newer toolkit or a subset with --cuda-archs")
    vs = args.vs_root
    if not vs:
        vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
        if vswhere.is_file():
            result = subprocess.run([str(vswhere), "-latest", "-prerelease", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"], capture_output=True, text=True, check=True)
            vs = Path(result.stdout.strip()) if result.stdout.strip() else None
    if not vs:
        raise ValueError("Visual Studio C++ tools not found; provide --vs-root")
    command = [str(ROOT / "tools" / "build_ggml_backend.cmd"), str(args.ggml_repo.resolve()), str(args.patched_source.resolve()), str(args.build_dir.resolve()), str(sdk), str(Path(args.cuda_root).resolve()), str(vs.resolve()), sys.executable, executable("cmake", args.cmake), executable("ninja", args.ninja)]
    print(f"Using Torch {torch.__version__} from {torch.__file__}; SDK: {sdk}", flush=True)
    environment = os.environ.copy()
    environment["COMFYUI_CUDA_ARCHS"] = ";".join(map(str, archs))
    subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError, KeyError) as exc:
        sys.exit(f"Build failed: {exc}")
