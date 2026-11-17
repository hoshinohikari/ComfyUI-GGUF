# ComfyUI-GGUF2: GGUF Quantization support for native ComfyUI models

This is currently very much WIP. These custom nodes provide support for model files stored in the GGUF format popularized by [llama.cpp](https://github.com/ggerganov/llama.cpp).

## Changes since leejet/HEAD

The comparison baseline is local `leejet/HEAD` at `373048b8403a7820620065210a691263d4da0a61`. The following changes have been committed on this branch; they have not been published as a release.

Committed changes:

- Detect MiniMax H3 32B text encoders and restore metadata for compatible MiniMax image GGUF models.
- Integrate file-backed GGUF weights with ComfyUI DynamicVRAM and its loading/offload paths.
- Add PyTorch dequantization for IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S and IQ3_XXS weights.

GGML backend and build changes (`cb4ff1e`):

- Add a pinned ggml CUDA MMQ backend for supported quantized Linear layers, retaining packed GGUF weights and using Q8_1 activations. It uses the active CUDA stream and a bounded workspace.
- Add Q2_0 reader/dequantization compatibility and extend the supported MMQ weight formats. Embeddings, normalization, floating-point layers and unsupported Linear cases follow their applicable existing paths.
- Add experimental `mixed` and `int8` Tensor Core modes, ggml as the loader default, and native build and verification scripts.

The Windows build now defaults to GeForce RTX 20/30/40/50 targets (SM75/86/89/120). This is build configuration support, not verification on all four generations. Some earlier GGML runs completed on an RTX 3060; other GPUs and the new multi-architecture build still require compilation and real-device verification. Python/Torch/runtime compatibility remains mandatory.

While quantization wasn't feasible for regular UNET models (conv2d), transformer/DiT models such as flux seem less affected by quantization. This allows running it in much lower bits per weight variable bitrate quants on low-end GPUs. For further VRAM savings, a node to load a quantized version of the T5 text encoder is also included.

![Comfy_Flux1_dev_Q4_0_GGUF_1024](https://github.com/user-attachments/assets/70d16d97-c522-4ef4-9435-633f128644c8)

Note: The "Force/Set CLIP Device" is **NOT** part of this node pack. Do not install it if you only have one GPU. Do not set it to cuda:0 then complain about OOM errors if you do not undestand what it is for. There is not need to copy the workflow above, just use your own workflow and replace the stock "Load Diffusion Model" with the "Unet Loader (GGUF)" node.

## Installation

### Windows one-click install (GitHub Release binary)

Close ComfyUI and double-click `install.cmd`. It discovers the ComfyUI source directory and its Python from the surrounding directories or standard Desktop install locations, then installs the matching Release extension. If multiple installations/Python environments are found, select one explicitly:

```powershell
.\install.cmd -ComfyUIPath "<ComfyUI source directory>" -Python "<ComfyUI Python executable>"
```

You can also download `install.ps1` alone and run it with those parameters; it clones this fork into `custom_nodes/ComfyUI-GGUF` if absent (Git required). Existing source checkouts are not reset or updated automatically. An old checkout lacking the installer needs a manual source update first. Portable Python and `.venv` paths are detected; Conda/system Python can be selected with `-Python`.

The installer checks Windows x64, CPython 3.13, CUDA Torch and the selected GPU, then searches stable Releases in `hoshinohikari/ComfyUI-GGUF` for a matching ZIP and SHA256. It validates the archive, native-source build ID, GPU target and extension metadata before replacing the binary, and backs up old extensions outside `native/build`. Plugin requirements are installed with `--no-deps` to preserve the host Torch; ComfyUI must already provide their transitive dependencies. No CUDA Toolkit or compiler is downloaded, and no generation or automatic restart is performed. Missing runtime DLLs or an unsupported environment produce an error; use `original` or build from source instead. Existing extension files are preserved if the pre-install checks fail.

Use `-ReleaseTag v0.0.1` to select a release. `-DryRun` checks environment and candidate assets without installing dependencies, downloading ZIPs or replacing files; full build-ID/DLL checks occur during installation. A precompiled extension still requires its matching source and compatible CUDA/VC++ runtime libraries.

> [!IMPORTANT]  
> Make sure your ComfyUI is on a recent-enough version to support custom ops when loading the UNET-only.

Clone this fork into `ComfyUI/custom_nodes` and install `requirements.txt` with the Python used by ComfyUI. Replace `<fork URL>` with this fork's published URL; the city96 upstream URL does not contain the changes described above.

```
git clone <fork URL> ComfyUI-GGUF
```

To install the custom node on a standalone ComfyUI release, open a CMD inside the "ComfyUI_windows_portable" folder (where your `run_nvidia_gpu.bat` file is) and use the following commands:

```
git clone <fork URL> ComfyUI/custom_nodes/ComfyUI-GGUF
.\python_embeded\python.exe -s -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-GGUF\requirements.txt
```

On MacOS sequoia, torch 2.4.1 seems to be required, as 2.6.X nightly versions cause a "M1 buffer is not large enough" error. See [this issue](https://github.com/city96/ComfyUI-GGUF/issues/107) for more information/workarounds.

## Usage

Simply use the GGUF Unet loader found under the `bootleg` category. Place the .gguf model files in your `ComfyUI/models/unet` folder.

The GGUF UNet and CLIP loaders expose an optional `compute_mode`:

- `mixed`: development mode; the experimental path uses W4A4 for Q4_K/IQ3_S and INT8 for Q5_K/Q6_K/IQ3_XXS.
- `int8`: development mode for supported GGUF Linear weights using INT8 Tensor Core operations.
- `ggml` (default): uses the pinned ggml CUDA MMQ backend for Q1_0/Q2_0/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, Q2_K–Q6_K, and IQ1_S/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS/IQ4_NL. It keeps the original GGUF packed weights and uses Q8_1 activations on RTX 3060. The plugin adds Q2_0 enum/size compatibility for gguf-py 0.19. The native extension must be built before selecting this mode.
- `original`: use the existing dequantize-then-PyTorch path.

The setting is per loader, so the diffusion model and text encoder can use different modes. New loader instances default to `ggml`; existing workflows keep any explicitly saved mode. `mixed` and `int8` dynamically quantize FP16, BF16, or FP32 activations; activation quantization can affect output quality. `ggml` converts bounded activation chunks to FP32 for the upstream MMQ entry, then quantizes them to Q8_1 internally. Unsupported per-layer shapes, patched weights, and unsupported operators use the original path. Selecting `ggml` with a missing or incompatible native extension currently reports an error: choose `original` manually in that case. Automatic fallback for an unavailable extension is planned, not implemented.

### Building the ggml extension on Windows

The known local toolchain is Windows x64, CPython 3.13, Torch `2.12.1+cu130`, CUDA Toolkit 13.5, MSVC, CMake and Ninja. Use the Python that runs your ComfyUI; do not replace its Torch merely to build this plugin. The toolkit must support every requested architecture; the Python wrapper checks the compiler's supported targets before building.

1. Install Git, the CUDA Toolkit and Visual Studio C++ build tools, and make CMake and Ninja available on `PATH`. Set `CUDA_PATH` to the toolkit root. `vswhere` discovers Visual Studio, including prerelease installations. The toolkit/compiler combination must itself be compatible.
2. From the plugin directory, initialize the ggml submodule pinned to commit `d9fe7bff3d2bf4d077e25c654486d02c321734aa`:

```powershell
git submodule update --init --recursive
```

3. Run the build using your ComfyUI Python. It uses `third_party/ggml` by default; supply `--ggml-repo` to use another checkout containing the pinned commit:

```powershell
& "<ComfyUI Python executable>" tools/build_native.py
```

The script discovers Torch headers and import libraries from that Python. It also accepts `--cuda-root`, `--vs-root`, `--cmake`, `--ninja`, `--build-dir` and `--patched-source` when automatic discovery or the defaults do not fit. Use `--help` for details. The output is placed in `native/build`; restart ComfyUI to load a newly built extension.

By default one extension contains SM75, SM86, SM89 and SM120 device code, covering GeForce RTX 20, 30, 40 and 50 respectively. The pinned ggml requires `120a-real` for its Blackwell instructions. To reduce compile time and binary size, build a subset:

```powershell
& "<ComfyUI Python executable>" tools/build_native.py --cuda-archs 86
& "<ComfyUI Python executable>" tools/build_native.py --cuda-archs 86,89,120
```

The legacy CMD entry reads the semicolon-separated `COMFYUI_CUDA_ARCHS` environment variable (default `75;86;89;120`). CMake uses `COMFYUI_CUDA_ARCHS` to generate real device-code targets; set that variable rather than `CMAKE_CUDA_ARCHITECTURES`. `build_info().cuda_archs` records the selected capabilities, and the loader checks the actual device against that list. No PTX forward-compatibility fallback is promised. Multi-architecture support does not remove the extension's Python/Torch/CUDA runtime requirements.

If the installed Torch lacks development files, extract them from a matching Torch wheel. The extractor's default SHA256 is for the local `torch-2.12.1+cu130-cp313-cp313-win_amd64.whl`; for another wheel, supply its independently verified `--sha256`. The wheel must match the active Torch version, Python and platform. The wrapper checks its SDK manifest; it does not install or replace Torch.

```powershell
& "<ComfyUI Python executable>" tools/extract_torch_sdk.py "<matching torch wheel>" "<SDK directory>" --sha256 "<wheel SHA256>"
& "<ComfyUI Python executable>" tools/build_native.py --torch-sdk "<SDK directory>\torch"
```

The build creates a detached worktree at the pinned ggml commit, applies the bridge there, builds the selected architectures and checks that the extension imports and reports the requested architecture list. It does not run a CUDA kernel. The legacy nine-argument `tools/build_ggml_backend.cmd` is still available. Build output is excluded from Git, and inference does not compile at runtime.

Rebuild after updating to this multi-architecture implementation: older SM86-only binaries do not satisfy the new source/metadata contract. Native source fingerprints normalize CRLF to LF so Git line-ending conversion alone no longer invalidates an otherwise matching build.

LoRA loading is experimental but it should work with just the built-in LoRA loader node(s).

Pre-quantized models:

- [flux1-dev GGUF](https://huggingface.co/city96/FLUX.1-dev-gguf)
- [flux1-schnell GGUF](https://huggingface.co/city96/FLUX.1-schnell-gguf)
- [stable-diffusion-3.5-large GGUF](https://huggingface.co/city96/stable-diffusion-3.5-large-gguf)
- [stable-diffusion-3.5-large-turbo GGUF](https://huggingface.co/city96/stable-diffusion-3.5-large-turbo-gguf)

Initial support for quantizing T5 has also been added recently, these can be used using the various `*CLIPLoader (gguf)` nodes which can be used inplace of the regular ones. For the CLIP model, use whatever model you were using before for CLIP. The loader can handle both types of files - `gguf` and regular `safetensors`/`bin`.

- [t5_v1.1-xxl GGUF](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf)

See the instructions in the [tools](https://github.com/city96/ComfyUI-GGUF/tree/main/tools) folder for how to create your own quants.
