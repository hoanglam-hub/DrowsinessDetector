@echo off
cd /d "%~dp0"
call .venv_runtime\Scripts\activate.bat
python tools\train_model.py --data-dir data\processed --models-dir models --sequence-length 30 --num-features 5 --epochs 120 --batch-size 32
pause
