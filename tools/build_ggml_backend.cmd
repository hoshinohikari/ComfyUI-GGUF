@echo off
setlocal

if "%~9"=="" goto usage

set "GGML_REPO=%~1"
set "PATCHED_SOURCE=%~2"
set "BUILD_DIR=%~3"
set "TORCH_SDK=%~4"
set "CUDA_ROOT=%~5"
set "VS_ROOT=%~6"
set "PYTHON_EXE=%~7"
set "CMAKE_EXE=%~8"
set "NINJA_EXE=%~9"
set "PLUGIN_ROOT=%~dp0.."
for %%I in ("%PLUGIN_ROOT%") do set "PLUGIN_ROOT=%%~fI"
set "PINNED_GGML=d9fe7bff3d2bf4d077e25c654486d02c321734aa"
set "CUDA_PATH=%CUDA_ROOT%"
set "CUDAToolkit_ROOT=%CUDA_ROOT%"
set "CUDACXX=%CUDA_ROOT%\bin\nvcc.exe"
set "PATH=%CUDA_ROOT%\bin;%PATH%"
if not defined COMFYUI_CUDA_ARCHS set "COMFYUI_CUDA_ARCHS=75;86;89;120"

if not exist "%GGML_REPO%\.git" (
    echo ERROR: ggml git checkout not found: %GGML_REPO%
    exit /b 2
)
if not exist "%TORCH_SDK%\share\cmake\Torch\TorchConfig.cmake" (
    echo ERROR: extracted Torch SDK not found: %TORCH_SDK%
    exit /b 2
)
if not exist "%CUDA_ROOT%\bin\nvcc.exe" (
    echo ERROR: CUDA compiler not found: %CUDA_ROOT%\bin\nvcc.exe
    exit /b 2
)
if not exist "%VS_ROOT%\VC\Auxiliary\Build\vcvars64.bat" (
    echo ERROR: vcvars64.bat not found under %VS_ROOT%
    exit /b 2
)

call "%VS_ROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if errorlevel 1 exit /b %errorlevel%

if not exist "%PATCHED_SOURCE%\CMakeLists.txt" (
    git -C "%GGML_REPO%" cat-file -e "%PINNED_GGML%^{commit}"
    if errorlevel 1 (
        echo ERROR: pinned ggml commit is absent from %GGML_REPO%
        exit /b 2
    )
    git -C "%GGML_REPO%" worktree add --detach "%PATCHED_SOURCE%" "%PINNED_GGML%"
    if errorlevel 1 exit /b %errorlevel%
)

"%PYTHON_EXE%" "%PLUGIN_ROOT%\tools\patch_ggml_source.py" "%PATCHED_SOURCE%"
if errorlevel 1 exit /b %errorlevel%

"%PYTHON_EXE%" "%PLUGIN_ROOT%\tools\native_build_id.py" > "%BUILD_DIR%\native-build-id.txt"
if errorlevel 1 exit /b %errorlevel%
set /p BUILD_ID=<"%BUILD_DIR%\native-build-id.txt"
if "%BUILD_ID%"=="" (
    echo ERROR: could not compute the native build ID
    exit /b 2
)
"%PYTHON_EXE%" -c "import torch; print(torch.__version__)" > "%BUILD_DIR%\torch-build-version.txt"
if errorlevel 1 exit /b %errorlevel%
set /p TORCH_BUILD=<"%BUILD_DIR%\torch-build-version.txt"
if "%TORCH_BUILD%"=="" (
    echo ERROR: could not read the active Torch version
    exit /b 2
)

"%CMAKE_EXE%" -S "%PLUGIN_ROOT%\native" -B "%BUILD_DIR%" -G Ninja ^
    -DCMAKE_MAKE_PROGRAM="%NINJA_EXE%" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CUDA_COMPILER="%CUDA_ROOT%\bin\nvcc.exe" ^
    "-DCOMFYUI_CUDA_ARCHS=%COMFYUI_CUDA_ARCHS%" ^
    -DCUDAToolkit_ROOT="%CUDA_ROOT%" ^
    -DPython3_EXECUTABLE="%PYTHON_EXE%" ^
    -DCOMFYUI_GGML_SOURCE="%PATCHED_SOURCE%" ^
    -DCOMFYUI_GGML_COMMIT="%PINNED_GGML%" ^
    -DTORCH_SDK_ROOT="%TORCH_SDK%" ^
    -DCOMFYUI_TORCH_BUILD="%TORCH_BUILD%" ^
    -DCOMFYUI_BUILD_ID="%BUILD_ID%" ^
    -DCOMFYUI_NATIVE_OUTPUT="%PLUGIN_ROOT%\native\build"
if errorlevel 1 exit /b %errorlevel%

"%CMAKE_EXE%" --build "%BUILD_DIR%" --target _comfyui_gguf_ggml --config Release --parallel %CMAKE_BUILD_PARALLEL_LEVEL%
if errorlevel 1 exit /b %errorlevel%

"%PYTHON_EXE%" "%PLUGIN_ROOT%\tools\verify_ggml_extension.py" "%PLUGIN_ROOT%\native\build" --build-id "%BUILD_ID%" --torch-build "%TORCH_BUILD%" --cuda-archs "%COMFYUI_CUDA_ARCHS%"
if errorlevel 1 exit /b %errorlevel%
exit /b 0

:usage
echo Usage:
echo   build_ggml_backend.cmd ^<ggml-repo^> ^<patched-source^> ^<build-dir^> ^<torch-sdk-root^> ^<cuda-root^> ^<vs-root^> ^<python.exe^> ^<cmake.exe^> ^<ninja.exe^>
echo Example paths are documented in README.md under the ggml native backend section.
exit /b 2
