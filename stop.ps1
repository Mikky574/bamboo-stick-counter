param([ValidateRange(1, 300)][int]$TimeoutSeconds = 15)
$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    Write-Error 'Missing .venv/Scripts/python.exe. Restore the local runtime before stopping a managed service.'
    exit 1
}
& $pythonPath -u (Join-Path $PSScriptRoot 'service_control.py') stop --timeout $TimeoutSeconds
exit $LASTEXITCODE
