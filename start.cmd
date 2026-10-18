@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
if errorlevel 1 (
    echo Start failed. See outputs\service.stderr.log and README.md.
    pause
    exit /b 1
)
echo Open http://127.0.0.1:5080/ in your browser.
