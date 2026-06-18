@echo off
cd /d "%~dp0"
call .venv_runtime\Scripts\activate.bat
python src\drowsiness_detector_tflite.py --camera 0
pause
