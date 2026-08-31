$ErrorActionPreference = "Stop"

$runtimeRoot = Join-Path (Split-Path -Parent $PSScriptRoot) ".runtime"
$processIds = @(
    Get-NetTCPConnection -State Listen -LocalPort 3000, 8000 -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
)
foreach ($name in @("web", "api")) {
    $pidFile = Join-Path $runtimeRoot "$name.pid"
    if (-not (Test-Path -LiteralPath $pidFile)) { continue }
    $processIds += [int](Get-Content -LiteralPath $pidFile)
    Remove-Item -LiteralPath $pidFile -Force
}
foreach ($processId in ($processIds | Select-Object -Unique)) {
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($process) { Stop-Process -Id $processId -ErrorAction SilentlyContinue }
}
Write-Host "VTLA Grip local services stopped."
