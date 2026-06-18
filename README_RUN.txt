1) Put videos here:
   data/normal/*.mp4
   data/drowsy/*.mp4

2) Create venv and install packages:
   setup_venv.bat

3) Extract features:
   run_extract.bat

4) Train model:
   run_train.bat

5) Run realtime app:
   run_app.bat

Outputs:
   models/drowsiness_gru.keras
   models/drowsiness_gru.tflite
   models/feature_scaler.joblib
   models/class_names.json
   models/classification_report.txt
