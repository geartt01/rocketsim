@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe -m rocketsim blender %*
if errorlevel 1 pause
