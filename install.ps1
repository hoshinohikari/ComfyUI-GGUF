param(
    [string]$ComfyUIPath,
    [string]$Python,
    [string]$ReleaseTag,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
try {
    if (-not $ComfyUIPath) {
        $candidates = @()
        foreach ($start in @($PSScriptRoot, (Get-Location).Path)) {
            $cursor = [IO.DirectoryInfo]$start
            while ($null -ne $cursor) {
                foreach ($candidate in @($cursor.FullName, (Join-Path $cursor.FullName 'ComfyUI'))) {
                    if ((Test-Path -LiteralPath (Join-Path $candidate 'main.py')) -and
                        (Test-Path -LiteralPath (Join-Path $candidate 'folder_paths.py'))) {
                        $candidates += [IO.Path]::GetFullPath($candidate)
                    }
                }
                $cursor = $cursor.Parent
            }
        }
        # Desktop can keep its install on a different drive from this checkout.
        $desktopRoots = @((Join-Path $env:LOCALAPPDATA 'Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI'))
        foreach ($drive in (Get-PSDrive -PSProvider FileSystem)) {
            $desktopRoots += Join-Path $drive.Root "Users\$env:USERNAME\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI"
        }
        foreach ($candidate in $desktopRoots) {
            if ((Test-Path -LiteralPath (Join-Path $candidate 'main.py')) -and
                (Test-Path -LiteralPath (Join-Path $candidate 'folder_paths.py'))) {
                $candidates += [IO.Path]::GetFullPath($candidate)
            }
        }
        $candidates = @($candidates | Select-Object -Unique)
        if ($candidates.Count -ne 1) {
            throw 'Cannot uniquely locate ComfyUI. Run install.cmd -ComfyUIPath "<ComfyUI directory>".'
        }
        $ComfyUIPath = $candidates[0]
    }
    $ComfyUIPath = (Resolve-Path -LiteralPath $ComfyUIPath).Path
    if (-not (Test-Path -LiteralPath (Join-Path $ComfyUIPath 'folder_paths.py'))) {
        throw 'ComfyUIPath must point to the ComfyUI source directory, not its portable parent.'
    }
    if (-not $Python) {
        $parent = Split-Path $ComfyUIPath -Parent
        $pythonCandidates = @(
            (Join-Path $ComfyUIPath '.venv\Scripts\python.exe'),
            (Join-Path $parent '.venv\Scripts\python.exe'),
            (Join-Path $parent 'python_embeded\python.exe'),
            (Join-Path $parent 'python_embedded\python.exe'),
            (Join-Path $ComfyUIPath 'venv\Scripts\python.exe')
        ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -Unique
        $pythonCandidates = @($pythonCandidates)
        if ($pythonCandidates.Count -ne 1) {
            throw 'Cannot uniquely locate ComfyUI Python. Supply -Python "<python.exe>" (also for Conda/system Python).'
        }
        $Python = $pythonCandidates[0]
    }
    $Python = (Resolve-Path -LiteralPath $Python).Path
    $plugin = Join-Path $ComfyUIPath 'custom_nodes\ComfyUI-GGUF'
    if (-not (Test-Path -LiteralPath $plugin)) {
        if ($DryRun) {
            Write-Host "Would clone https://github.com/hoshinohikari/ComfyUI-GGUF.git into $plugin"
            return
        }
        & git clone https://github.com/hoshinohikari/ComfyUI-GGUF.git $plugin
        if ($LASTEXITCODE -ne 0) { throw 'git clone failed; install Git or clone the project manually.' }
    }
    $installer = Join-Path $plugin 'tools\install_release.py'
    if (-not (Test-Path -LiteralPath $installer)) {
        throw "Existing plugin lacks tools/install_release.py. Update this fork manually first: $plugin"
    }
    $installerArgs = @('-s', $installer)
    if ($ReleaseTag) { $installerArgs += @('--release', $ReleaseTag) }
    if ($DryRun) { $installerArgs += '--dry-run' }
    & $Python @installerArgs
    if ($LASTEXITCODE -ne 0) { throw "Installer failed with exit code $LASTEXITCODE" }
} catch {
    Write-Host "Installation failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
