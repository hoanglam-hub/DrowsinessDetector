@echo off
cd /d "%~dp0"
call .venv_runtime\Scripts\activate.bat
python tools\extract_to_csv.py --data-dir data --out-dir data\processed --sequence-length 30 --stride 10 --resize-width 640
pause
