param(
    [string]$Proxy = '',
    [string]$PythonCommand = 'python',
    [string]$ModelArchiveDirectory = ''
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
function Check-LastExit {
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE). Existing environments and models were retained." }
}
& $PythonCommand -c 'import sys; assert (3,11) <= sys.version_info[:2] <= (3,12), "Use Python 3.11 or 3.12 for the tested Torch 2.6 runtime."'
Check-LastExit
$webRuntime = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$runtime = Join-Path $PSScriptRoot '.venv-countgdpp/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $webRuntime)) {
    & $PythonCommand -m venv .venv
    Check-LastExit
}
if (-not (Test-Path -LiteralPath $runtime)) {
    & $PythonCommand -m venv .venv-countgdpp
    Check-LastExit
}
$pipOptions = @('--index-url', 'https://pypi.org/simple')
$proxyOptions = @()
if ($Proxy) {
    $pipOptions += @('--proxy', $Proxy)
    $proxyOptions = @('--proxy', $Proxy)
}
& $webRuntime -m pip install -r requirements.txt @pipOptions
Check-LastExit
& $runtime -c 'import torch, torchvision; assert torch.__version__.startswith("2.6.") and torchvision.__version__.startswith("0.21.") and torch.version.cuda, "CUDA Torch 2.6/torchvision 0.21 required"'
if ($LASTEXITCODE -ne 0) {
    & $runtime -m pip install --upgrade --force-reinstall torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126 @proxyOptions
    Check-LastExit
}
& $runtime -m pip install -r requirements-auto.txt @pipOptions
Check-LastExit
$vendor = Join-Path $PSScriptRoot 'vendor/countgdpp'
$commit = '8489f836d75eb407ae778c56cff1f72cc6f8baae'
$gitOptions = @('-c', 'http.proxy=')
if ($Proxy) { $gitOptions = @('-c', "http.proxy=$Proxy") }
if (-not (Test-Path -LiteralPath $vendor)) {
    New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot 'vendor') | Out-Null
    git @gitOptions clone --filter=blob:none --no-checkout https://github.com/niki-amini-naieni/CountGDPlusPlus.git $vendor
    Check-LastExit
    git @gitOptions -C $vendor checkout --detach $commit
    Check-LastExit
}
$actualCommit = git -C $vendor rev-parse HEAD
Check-LastExit
if ($actualCommit.Trim() -ne $commit) {
    throw 'Vendor revision differs from the tested version; existing checkout was preserved. Use the pinned source before continuing.'
}
& $runtime prepare_countgdpp.py --patch-only
Check-LastExit
$restoreOptions = @('--tag', 'v0.2.0') + $proxyOptions
if ($ModelArchiveDirectory) { $restoreOptions += @('--archive-dir', (Resolve-Path -LiteralPath $ModelArchiveDirectory).Path) }
& $webRuntime tools/fetch_models.py @restoreOptions
Check-LastExit
Write-Output 'Setup complete: web environment, CUDA inference environment, pinned patched vendor, CountGD++ and active ConvNeXt-Small v2 assets are installed.'
Write-Output 'No inference or server was started. Run .\start.ps1 when ready. The service prewarms and retains its models until .\stop.ps1.'
