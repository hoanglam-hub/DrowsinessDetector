@echo off
cd /d "%~dp0"
python -m venv .venv_runtime
call .venv_runtime\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
pause
