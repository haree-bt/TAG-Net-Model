@echo off
setlocal
REM ==============================================================================
REM TAG-Net Real-Time Interactive MuJoCo 3D Simulation Launcher
REM Usage:
REM   run_mujoco_sim.bat [knife|mug|tool] [handover|pick_and_place|work_with_it] [tag_net|baseline] [orientation_deg]
REM Examples:
REM   run_mujoco_sim.bat knife handover tag_net 0
REM   run_mujoco_sim.bat knife handover baseline 0
REM   run_mujoco_sim.bat mug pick_and_place tag_net 45
REM ==============================================================================

set TOOL=%1
set TASK=%2
set MODEL=%3
set ORIENTATION=%4

if "%TOOL%"=="" set TOOL=knife
if "%TASK%"=="" set TASK=handover
if "%MODEL%"=="" set MODEL=tag_net
if "%ORIENTATION%"=="" set ORIENTATION=0.0

if /i "%TOOL%"=="tool" set TOOL=screwdriver

echo ==================================================================
echo [TAG-Net] Launching MuJoCo 3D Simulation
echo   - Object      : %TOOL%
echo   - Task        : %TASK%
echo   - Model       : %MODEL%
echo   - Orientation : %ORIENTATION% deg
echo ==================================================================

cd /d "C:\Users\hareh\.gemini\antigravity\scratch\tag_grasp_net"
python simulation/realtime_mujoco_simulation.py --tool %TOOL% --task %TASK% --model %MODEL% --orientation %ORIENTATION%
pause
