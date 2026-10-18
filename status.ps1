$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    Write-Error 'Missing .venv/Scripts/python.exe. Complete the README installation first.'
    exit 1
}
& $pythonPath -u (Join-Path $PSScriptRoot 'service_control.py') status
exit $LASTEXITCODE
