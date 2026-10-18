param(
    [ValidateRange(1, 3600)][int]$TimeoutSeconds = 90,
    [ValidateRange(1, 65535)][int]$Port = 5080
)
$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    Write-Error 'Missing .venv/Scripts/python.exe. Complete the README installation first.'
    exit 1
}
& $pythonPath -u (Join-Path $PSScriptRoot 'service_control.py') start --port $Port --timeout $TimeoutSeconds
exit $LASTEXITCODE
