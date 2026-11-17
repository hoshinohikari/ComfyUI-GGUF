"""Install a matching GitHub Release extension into this ComfyUI-GGUF checkout."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile

from native_build_id import compute_build_id

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'hoshinohikari/ComfyUI-GGUF'
BINARY = '_comfyui_gguf_ggml.cp313-win_amd64.pyd'
LIMIT = 512 * 1024 * 1024


def request(url: str):
    if urllib.parse.urlparse(url).scheme != 'https':
        raise ValueError('Release downloads must use HTTPS')
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        'User-Agent': 'ComfyUI-GGUF-installer',
    }), timeout=60)


def download(url: str, destination: Path) -> None:
    with request(url) as source, destination.open('wb') as target:
        size = 0
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            if size > LIMIT:
                raise ValueError('Release asset exceeds download limit')
            target.write(chunk)


def asset_candidates(releases: list, torch_version: str) -> list:
    prefix = f'comfyui-gguf-ggml-cp313-win_amd64-torch{torch_version.replace("+", "-")}-'
    result = []
    for release in releases:
        if release.get('draft') or release.get('prerelease'):
            continue
        assets = {a['name']: a for a in release.get('assets', [])}
        for name, asset in assets.items():
            if name.startswith(prefix) and name.endswith('.zip') and name + '.sha256' in assets:
                result.append((release['tag_name'], asset, assets[name + '.sha256']))
    return result


def inspect_archive(path: Path, torch_version: str, arch: int, build_id: str) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP members')
        name = f'native/build/{BINARY}'
        for required in ('build-manifest.json', name):
            if required not in names:
                raise ValueError(f'Missing {required}')
        if archive.getinfo('build-manifest.json').file_size > 1024 * 1024 or archive.getinfo(name).file_size > LIMIT:
            raise ValueError('Release contents exceed size limit')
        manifest = json.loads(archive.read('build-manifest.json'))
        info = manifest['build_info']
        if info.get('torch_build') != torch_version or info.get('build_id') != build_id:
            raise ValueError('Release Torch/build ID does not match installed source')
        if arch not in info.get('cuda_archs', []):
            raise ValueError(f'Release does not contain SM{arch}')
        if hashlib.sha256(archive.read(name)).hexdigest() != manifest['extension_sha256']:
            raise ValueError('Extension SHA256 mismatch')
        return manifest


def verify_extension(path: Path, expected: dict) -> None:
    # Use a separate process so its DLL handle closes before replacement on Windows.
    code = '''
import importlib.util,json,os,pathlib,sys,torch
p=pathlib.Path(sys.argv[1])
handles=[os.add_dll_directory(str(pathlib.Path(torch.__file__).parent/'lib'))]
if os.environ.get('CUDA_PATH'):
    handles.append(os.add_dll_directory(str(pathlib.Path(os.environ['CUDA_PATH'])/'bin')))
s=importlib.util.spec_from_file_location('_comfyui_gguf_ggml',p)
m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
print(json.dumps(m.build_info()))
'''
    output = subprocess.check_output([sys.executable, '-s', '-c', code, str(path)], text=True)
    info = json.loads(output.splitlines()[-1])
    for field in ('build_id', 'torch_build', 'ggml_commit', 'cuda_archs', 'weight_types'):
        if info.get(field) != expected.get(field):
            raise ValueError(f'Loaded extension metadata differs from manifest: {field}')


def install_binary(source: Path, manifest: dict) -> None:
    target = ROOT / 'native' / 'build'
    target.mkdir(parents=True, exist_ok=True)
    staging = target / (BINARY + '.new')
    shutil.copy2(source, staging)
    # Keep backups outside native/build so the loader still finds exactly one .pyd.
    backup = ROOT / 'build' / 'installer-backups' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    old = list(target.glob('_comfyui_gguf_ggml*.pyd'))
    moved = []
    installed = False
    try:
        if old:
            backup.mkdir(parents=True)
        for item in old:
            item.rename(backup / item.name)
            moved.append(item.name)
        staging.replace(target / BINARY)
        installed = True
        (target / 'release-build-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    except BaseException:
        if installed:
            (target / BINARY).unlink()
        for name in moved:
            (backup / name).rename(target / name)
        raise
    finally:
        staging.unlink(missing_ok=True)
    if moved:
        print(f'Previous extension backed up to {backup}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', help='Release tag; otherwise search recent stable releases')
    parser.add_argument('--dry-run', action='store_true', help='Inspect environment and assets without downloads or changes')
    args = parser.parse_args()
    if sys.platform != 'win32' or platform.machine().lower() not in ('amd64', 'x86_64') or sys.version_info[:2] != (3, 13):
        raise ValueError('Current Release packages require Windows x64 / CPython 3.13')
    import torch
    if not torch.version.cuda or not torch.cuda.is_available():
        raise ValueError('This Python needs CUDA-enabled Torch and an available NVIDIA GPU; check the ComfyUI Python and driver')
    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    arch = major * 10 + minor
    print(f'Python: {sys.executable}\nTorch: {torch.__version__} (CUDA {torch.version.cuda})\nGPU: {torch.cuda.get_device_name(device)} / SM{arch}')
    endpoint = f'https://api.github.com/repos/{REPOSITORY}/releases'
    if args.release:
        endpoint += '/tags/' + urllib.parse.quote(args.release, safe='')
    else:
        endpoint += '?per_page=30'
    with request(endpoint) as response:
        releases = json.load(response)
    candidates = asset_candidates([releases] if args.release else releases, torch.__version__)
    if not candidates:
        raise ValueError('No matching ZIP + SHA256 in Releases. No Torch changes made; use original mode or build from source')
    if args.dry_run:
        print('Candidate releases: ' + ', '.join(tag + '/' + a['name'] for tag, a, _ in candidates))
        print('Dry run: manifest/build ID and DLL load still require downloading the package')
        return
    print('Close ComfyUI before replacing its native extension.')
    with tempfile.TemporaryDirectory(prefix='comfyui-gguf-install-') as temporary:
        temp = Path(temporary)
        selected = None
        for tag, asset, checksum_asset in candidates:
            archive = temp / 'release.zip'
            checksum = temp / 'release.sha256'
            print(f'Downloading {tag}/{asset["name"]}')
            download(asset['browser_download_url'], archive)
            download(checksum_asset['browser_download_url'], checksum)
            expected_hash = checksum.read_text(encoding='utf-8-sig').split()[0]
            if not re.fullmatch(r'[a-fA-F0-9]{64}', expected_hash) or hashlib.sha256(archive.read_bytes()).hexdigest() != expected_hash.lower():
                raise ValueError('Release ZIP SHA256 mismatch')
            try:
                manifest = inspect_archive(archive, torch.__version__, arch, compute_build_id())
            except ValueError as exc:
                print(f'Skipping incompatible package: {exc}')
                continue
            with zipfile.ZipFile(archive) as z:
                # Read a single known member; never extract arbitrary paths from a ZIP.
                extension = temp / BINARY
                extension.write_bytes(z.read(f'native/build/{BINARY}'))
            selected = extension, manifest
            break
        if selected is None:
            raise ValueError('Release packages do not match this source/GPU. Update this fork manually or build from source')
        subprocess.run([sys.executable, '-s', '-m', 'pip', 'install', '--no-deps', '-r', str(ROOT / 'requirements.txt')], check=True)
        verify_extension(selected[0], selected[1]['build_info'])
        install_binary(*selected)
    print('Installed. Restart ComfyUI and select ggml. No GPU kernel/generation test was run.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, ImportError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        sys.exit(f'Installation failed: {exc}')
