# PowerShell launcher for AL-QNN Quantum Studio
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  KHOI DONG AL-QNN QUANTUM STUDIO (UI/UX DASHBOARD)" -ForegroundColor Green
Write-Host "  Dia chi: http://localhost:8501" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Cyan

$StudioDir = Join-Path $PSScriptRoot "al_qnn_studio"
Set-Location $StudioDir

Start-Process "http://localhost:8501"
python -m http.server 8501
