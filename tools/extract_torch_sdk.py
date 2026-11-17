"""Extract only the development SDK from the matching PyTorch wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


EXPECTED_SHA256 = "76dd848312a40d29499614b714a4318841734ff309ad922f2365868395b6b054"
WHEEL_NAME = "torch-2.12.1+cu130-cp313-cp313-win_amd64.whl"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_sdk_file(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        return False
    return (
        name.startswith("torch/include/")
        or name.startswith("torch/share/cmake/")
        or (name.startswith("torch/lib/") and name.endswith(".lib"))
    )


def extract_sdk(wheel: Path, destination: Path, expected_sha256: str) -> int:
    actual_sha256 = sha256(wheel)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"Wheel SHA256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            if member.is_dir() or not is_sdk_file(member.filename):
                continue
            relative = PurePosixPath(member.filename)
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                output.write(source.read())
            extracted.append(member.filename)

    required = {
        "torch/include/torch/extension.h",
        "torch/include/torch/csrc/stable/library.h",
        "torch/include/torch/csrc/stable/tensor.h",
        "torch/lib/torch.lib",
        "torch/lib/torch_cpu.lib",
        "torch/lib/torch_cuda.lib",
        "torch/lib/c10.lib",
        "torch/lib/c10_cuda.lib",
        "torch/share/cmake/Torch/TorchConfig.cmake",
    }
    missing = sorted(required - set(extracted))
    if missing:
        raise ValueError("Wheel is missing required SDK files: " + ", ".join(missing))

    manifest = {
        "wheel": wheel.name,
        "sha256": actual_sha256,
        "files": len(extracted),
        "bytes": sum((destination / Path(*PurePosixPath(name).parts)).stat().st_size for name in extracted),
    }
    (destination / "torch-sdk-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return len(extracted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--sha256", default=EXPECTED_SHA256)
    args = parser.parse_args()
    count = extract_sdk(args.wheel, args.destination, args.sha256)
    print(f"Extracted {count} SDK files to {args.destination.resolve()}")


if __name__ == "__main__":
    main()
