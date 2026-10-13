@echo off
title TAG-Net Grasp Synthesis API
echo ======================================================================
echo    TAG-Net: Task-Aware Robotic Grasping API Server
echo    FastAPI + Uvicorn Service with Interactive Swagger & Web Studio
echo ======================================================================
echo.

:: Free port 8000 if already occupied by an old instance
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo Freeing port 8000 from old process %%a...
    taskkill /F /PID %%a >nul 2>&1
)

python api.py
pause
