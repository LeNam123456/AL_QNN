param (
    [string]$Dataset = "predictive_maintenance"
)

Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "   KHOI DONG LIT DEMO CHO MO HINH AL-QNN               " -ForegroundColor Yellow
Write-Host "=======================================================" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

if ($Dataset -eq "heart") {
    Write-Host "Dang mo dataset Heart Disease tren http://localhost:7002 ..." -ForegroundColor Green
    python lit_demo.py heart
} else {
    Write-Host "Dang mo dataset Predictive Maintenance tren http://localhost:7001 ..." -ForegroundColor Green
    python lit_demo.py predictive_maintenance
}
