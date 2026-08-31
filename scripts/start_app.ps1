$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$frontendRoot = Join-Path $projectRoot "frontend"
$runtimeRoot = Join-Path $projectRoot ".runtime"
New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null

function Test-LocalPort([int]$Port) {
    return [bool](Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
}

if (-not (Test-LocalPort 8000)) {
    $uv = (Get-Command uv -ErrorAction Stop).Source
    $api = Start-Process -FilePath $uv -ArgumentList @("run", "vtla-api") -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru
    Set-Content -LiteralPath (Join-Path $runtimeRoot "api.pid") -Value $api.Id
}

if (-not (Test-LocalPort 3000)) {
    $vinextLock = Join-Path $frontendRoot ".vinext\dev\lock.json"
    if (Test-Path -LiteralPath $vinextLock) { Remove-Item -LiteralPath $vinextLock -Force }
    $nodeDirectory = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages\OpenJS.NodeJS.LTS_Microsoft.Winget.Source_8wekyb3d8bbwe\node-v24.19.0-win-x64"
    if (Test-Path -LiteralPath $nodeDirectory) { $env:Path = "$nodeDirectory;$env:Path" }
    $npm = (Get-Command npm.cmd -ErrorAction Stop).Source
    $web = Start-Process -FilePath $npm -ArgumentList @("run", "dev") -WorkingDirectory $frontendRoot -WindowStyle Hidden -PassThru
    Set-Content -LiteralPath (Join-Path $runtimeRoot "web.pid") -Value $web.Id
}

$deadline = (Get-Date).AddSeconds(20)
while (-not (Test-LocalPort 3000) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 250 }
if (-not (Test-LocalPort 3000)) { throw "Frontend did not start within 20 seconds. Run npm run dev in frontend for details." }

Start-Process "http://localhost:3000"
Write-Host "VTLA Grip started: http://localhost:3000"
Write-Host "Hardware API is loopback-only: http://127.0.0.1:8000"
