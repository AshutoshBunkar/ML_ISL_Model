@echo off
title Indian Sign Language Recognition System
cd /d "%~dp0"
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe run.py
) else (
    python run.py
)
pause
