@echo off
setlocal
REM ==============================================================================
REM TAG-Net Real-Time Interactive MuJoCo 3D Simulation Launcher
REM Usage:
REM   run_mujoco_sim.bat [knife|mug|tool] [pick_and_place|handover|work_with_it]
REM Examples:
REM   run_mujoco_sim.bat mug
REM   run_mujoco_sim.bat tool work_with_it
REM   run_mujoco_sim.bat knife pick_and_place
REM ==============================================================================

set TOOL=%1
set TASK=%2

if "%TOOL%"=="" set TOOL=knife
if "%TASK%"=="" set TASK=handover

if /i "%TOOL%"=="tool" set TOOL=screwdriver

echo ==================================================================
echo [TAG-Net] Launching MuJoCo 3D Simulation
echo   - Object : %TOOL%
echo   - Task   : %TASK%
echo ==================================================================

cd /d "C:\Users\hareh\.gemini\antigravity\scratch\tag_grasp_net"
python simulation/realtime_mujoco_simulation.py --tool %TOOL% --task %TASK%
pause
