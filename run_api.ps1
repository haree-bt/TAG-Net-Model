# TAG-Net FastAPI Service PowerShell Launcher

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   TAG-Net: Task-Aware Robotic Grasping API Server" -ForegroundColor Green
Write-Host "   FastAPI + Uvicorn Service with Interactive Swagger & Web Studio" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Free port 8000 if already occupied by an old instance
$conn = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Where-Object { $_.State -eq 'Listen' }
if ($conn) {
    Write-Host "Freeing port 8000 from existing PID $($conn.OwningProcess)..." -ForegroundColor Yellow
    Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "Launching FastAPI service on http://localhost:8000 ..." -ForegroundColor Cyan
python api.py
