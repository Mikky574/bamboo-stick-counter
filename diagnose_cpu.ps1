param(
    [string]$Image = '',
    [string]$Category = 'mung',
    [string]$Text = '',
    [switch]$RunModel,
    [switch]$CheckScene,
    [ValidateRange(1, 3600)][int]$TimeoutSeconds = 600,
    [string]$Python = ''
)
$ErrorActionPreference = 'Stop'
$modelPython = $Python
if (-not $modelPython) {
    $modelPython = Join-Path $PSScriptRoot '.venv-countgdpp/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $modelPython)) {
        $fallbackPython = Get-Command python -ErrorAction SilentlyContinue
        if (-not $fallbackPython) { throw 'Python not found. Install Python 3.11/3.12 or pass -Python with its full path.' }
        $modelPython = $fallbackPython.Source
        Write-Output 'Model environment is missing; inspecting the current Python. Missing dependencies will be reported.'
    }
}
$modelPython = (Resolve-Path -LiteralPath $modelPython).Path
$probeArguments = @('-u', (Join-Path $PSScriptRoot 'tools/diagnose_cpu.py'), '--python', $modelPython, '--timeout', "$TimeoutSeconds")
if ($Image) { $probeArguments += @('--image', (Resolve-Path -LiteralPath $Image).Path) }
if ($Category) { $probeArguments += @('--category', $Category) }
if ($PSBoundParameters.ContainsKey('Text')) { $probeArguments += @('--text', $Text) }
if ($RunModel) { $probeArguments += '--run-model' }
if ($CheckScene) { $probeArguments += '--check-scene' }
& $modelPython @probeArguments
exit $LASTEXITCODE
