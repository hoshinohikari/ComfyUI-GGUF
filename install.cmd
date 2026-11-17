@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
set "INSTALL_RESULT=%ERRORLEVEL%"
if not "%COMFYUI_GGUF_NO_PAUSE%"=="1" pause
exit /b %INSTALL_RESULT%
