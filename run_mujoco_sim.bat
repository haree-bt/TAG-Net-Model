@echo off
REM ==============================================================================
REM TAG-Net Real-Time Interactive MuJoCo 3D Simulation Launcher
REM Usage:
REM   run_mujoco_sim.bat knife
REM   run_mujoco_sim.bat mug
REM   run_mujoco_sim.bat screwdriver
REM ==============================================================================

set TOOL=%1
if "%TOOL%"=="" set TOOL=knife

echo ==================================================================
echo [TAG-Net] Starting Real-Time Interactive MuJoCo Simulation: %TOOL%
echo ==================================================================

cd /d "C:\Users\hareh\.gemini\antigravity\scratch\tag_grasp_net"
python simulation/realtime_mujoco_simulation.py --tool %TOOL%
pause
