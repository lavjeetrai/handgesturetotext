@echo off
setlocal
cd /d "%~dp0"

if exist "..\.venv\Scripts\python.exe" (
    "..\.venv\Scripts\python.exe" final_pred.py
) else (
    python final_pred.py
)

pause
