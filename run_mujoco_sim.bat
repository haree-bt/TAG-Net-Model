@echo off
setlocal
REM ==============================================================================
REM TAG-Net Real-Time Interactive MuJoCo 3D Simulation Launcher
REM Usage:
REM   run_mujoco_sim.bat knife
REM   run_mujoco_sim.bat mug
REM   run_mujoco_sim.bat tool (or screwdriver)
REM   run_mujoco_sim.bat (opens interactive menu)
REM ==============================================================================

set TOOL=%1

if "%TOOL%"=="" (
    echo ==================================================================
    echo       TAG-Net Real-Time Interactive MuJoCo 3D Simulation
    echo ==================================================================
    echo Select object to simulate:
    echo   [1] Knife        (Avoids cutting blade, grasps handle, hands over safely)
    echo   [2] Ceramic Mug  (Clears drinking rim, grasps lateral handle, hands over)
    echo   [3] Tool/Driver  (Clears metal shaft, grasps rubberized grip, hands over)
    echo ==================================================================
    choice /c 123 /m "Enter choice (1, 2, or 3): "
    if errorlevel 3 (
        set TOOL=screwdriver
    ) else if errorlevel 2 (
        set TOOL=mug
    ) else (
        set TOOL=knife
    )
)

if /i "%TOOL%"=="tool" set TOOL=screwdriver

echo.
echo ==================================================================
echo [TAG-Net] Launching MuJoCo Real-Time 3D Simulation for: %TOOL%
echo ==================================================================

cd /d "C:\Users\hareh\.gemini\antigravity\scratch\tag_grasp_net"
python simulation/realtime_mujoco_simulation.py --tool %TOOL%
pause
