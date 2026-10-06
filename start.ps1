Set-Location $PSScriptRoot
$pythonPath = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (Test-Path -LiteralPath $pythonPath) {
    & $pythonPath app.py
} else {
    python app.py
}
