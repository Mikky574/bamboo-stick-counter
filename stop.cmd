@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop.ps1" %*
if errorlevel 1 (
    echo Stop failed. See outputs\service.stderr.log and README.md.
    pause
    exit /b 1
)
