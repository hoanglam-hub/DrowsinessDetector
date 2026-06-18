"""
drowsiness_detector_tflite_runtime.py

Runtime phát hiện ngủ gật bằng Hybrid:
    MediaPipe FaceMesh
    -> Feature: ear_L, ear_R, mar, pitch, yaw
    -> feature_scaler.joblib
    -> drowsiness_gru.tflite
    -> Rule EAR/MAR/Head Pose/No Face
    -> 3 mức cảnh báo

Yêu cầu thư mục:
    models/drowsiness_gru.tflite
    models/feature_scaler.joblib

    audios/alert_lv1.wav
    audios/alert_lv2.wav
    audios/alert_lv3.wav

Chạy:
    python src/drowsiness_detector_tflite_runtime.py --camera 0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import cv2
import joblib
import numpy as np
import pygame


# ============================================================
# PATH
# ============================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]


# ============================================================
# TFLITE INTERPRETER
# ============================================================

try:
    from tflite_runtime.interpreter import Interpreter
    print("[INFO] Using tflite-runtime Interpreter")
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
        print("[INFO] Using tensorflow-cpu TFLite Interpreter")
    except ImportError as e:
        raise ImportError(
            "Không tìm thấy TFLite Interpreter.\n"
            "Cài một trong hai cách:\n"
            "  pip install tflite-runtime\n"
            "hoặc:\n"
            "  pip install tensorflow-cpu==2.18.1"
        ) from e


# ============================================================
# MEDIAPIPE FACE MESH
# ============================================================

try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
except AttributeError:
    from mediapipe.python.solutions import face_mesh as mp_face_mesh


# ============================================================
# FEATURE UTILS - NHÚNG TRỰC TIẾP TRONG FILE RUNTIME
# ============================================================

def _distance(p1, p2) -> float:
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    return float(np.linalg.norm(p1 - p2))


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    if abs(float(b)) < 1e-8:
        return default
    return float(a) / float(b)


def calculate_ear(eye_points) -> float:
    """
    EAR - Eye Aspect Ratio.
    Dùng để xác định mắt nhắm/mở.
    """
    A = _distance(eye_points[1], eye_points[5])
    B = _distance(eye_points[2], eye_points[4])
    C = _distance(eye_points[0], eye_points[3])
    return _safe_div(A + B, 2.0 * C, default=0.0)


def calculate_mar(mouth_points) -> float:
    """
    MAR - Mouth Aspect Ratio.
    Dùng để xác định ngáp.
    """
    A = _distance(mouth_points[1], mouth_points[7])
    B = _distance(mouth_points[2], mouth_points[6])
    C = _distance(mouth_points[3], mouth_points[5])
    D = _distance(mouth_points[0], mouth_points[4])
    return _safe_div(A + B + C, 2.0 * D, default=0.0)


def get_head_pose_simple(landmarks, frame_shape) -> Tuple[float, float]:
    """
    Tính pitch/yaw đơn giản, giữ đúng logic train cũ:
        pitch = khoảng mũi-cằm / khoảng 2 mắt
        yaw   = khoảng mũi-mắt trái / khoảng 2 mắt - 0.5
    """
    h, w = frame_shape[:2]

    nose_tip = [landmarks[1].x * w, landmarks[1].y * h]
    chin = [landmarks[152].x * w, landmarks[152].y * h]
    left_eye = [landmarks[33].x * w, landmarks[33].y * h]
    right_eye = [landmarks[263].x * w, landmarks[263].y * h]

    face_width = _distance(left_eye, right_eye)
    nose_to_left = _distance(nose_tip, left_eye)
    face_height = _distance(nose_tip, chin)

    yaw = _safe_div(nose_to_left, face_width, default=0.5) - 0.5
    pitch = _safe_div(face_height, face_width, default=0.0)

    return float(pitch), float(yaw)


def extract_all_features(landmarks, frame_shape) -> Optional[np.ndarray]:
    """
    Thứ tự feature phải đúng với lúc train:
        [ear_L, ear_R, mar, pitch, yaw]
    """
    left_eye_indices = [33, 160, 158, 133, 153, 144]
    right_eye_indices = [362, 385, 387, 263, 373, 380]
    mouth_indices = [78, 81, 13, 311, 308, 402, 14, 178]

    h, w = frame_shape[:2]

    left_eye = [(landmarks[i].x * w, landmarks[i].y * h) for i in left_eye_indices]
    right_eye = [(landmarks[i].x * w, landmarks[i].y * h) for i in right_eye_indices]
    mouth = [(landmarks[i].x * w, landmarks[i].y * h) for i in mouth_indices]

    ear_l = calculate_ear(left_eye)
    ear_r = calculate_ear(right_eye)
    mar = calculate_mar(mouth)
    pitch, yaw = get_head_pose_simple(landmarks, frame_shape)

    features = np.asarray([ear_l, ear_r, mar, pitch, yaw], dtype=np.float32)

    if not np.all(np.isfinite(features)):
        return None

    return features


# ============================================================
# MAIN DETECTOR
# ============================================================

class DrowsinessDetectorTFLiteRuntime:
    def __init__(
        self,
        model_path: str | Path = PROJECT_ROOT / "models" / "drowsiness_gru.tflite",
        scaler_path: str | Path = PROJECT_ROOT / "models" / "feature_scaler.joblib",
        audio_level1_path: str | Path = PROJECT_ROOT / "audios" / "alert_lv1.wav",
        audio_level2_path: str | Path = PROJECT_ROOT / "audios" / "alert_lv2.wav",
        audio_level3_path: str | Path = PROJECT_ROOT / "audios" / "alert_lv3.wav",

        sequence_length: int = 30,
        num_features: int = 5,

        # Model
        model_threshold: float = 0.85,
        model_smooth_window: int = 10,
        model_consecutive_level1: int = 2,
        model_consecutive_level2: int = 5,
        model_consecutive_level3: int = 9,

        # Rule mắt
        ear_closed: float = 0.20,
        ear_low: float = 0.24,

        # Rule ngáp
        mar_yawn: float = 0.65,

        # Rule cúi/gật đầu
        pitch_delta: float = 0.18,

        # Thời gian rule riêng cho 3 mức
        level1_eye_low_seconds: float = 1.2,
        level1_eye_closed_seconds: float = 0.8,
        level1_yawn_seconds: float = 1.0,

        level2_eye_closed_seconds: float = 2.0,
        level2_yawn_seconds: float = 2.0,
        level2_combined_seconds: float = 1.5,

        level3_eye_closed_seconds: float = 4.0,
        level3_no_face_seconds: float = 6.0,

        # Mất mặt
        no_face_level1_seconds: float = 2.5,
        no_face_level2_seconds: float = 4.0,

        # Âm thanh
        audio_cooldown: float = 3.0,

        # Camera
        resize_width: int = 640,
        camera_width: int = 640,
        camera_height: int = 480,

        # Mediapipe
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)

        self.audio_paths = {
            1: Path(audio_level1_path),
            2: Path(audio_level2_path),
            3: Path(audio_level3_path),
        }

        self.sequence_length = int(sequence_length)
        self.num_features = int(num_features)

        self.model_threshold = float(model_threshold)
        self.model_smooth_window = int(model_smooth_window)

        self.model_consecutive_level1 = int(model_consecutive_level1)
        self.model_consecutive_level2 = int(model_consecutive_level2)
        self.model_consecutive_level3 = int(model_consecutive_level3)

        self.ear_closed = float(ear_closed)
        self.ear_low = float(ear_low)
        self.mar_yawn = float(mar_yawn)
        self.pitch_delta = float(pitch_delta)

        self.level1_eye_low_seconds = float(level1_eye_low_seconds)
        self.level1_eye_closed_seconds = float(level1_eye_closed_seconds)
        self.level1_yawn_seconds = float(level1_yawn_seconds)

        self.level2_eye_closed_seconds = float(level2_eye_closed_seconds)
        self.level2_yawn_seconds = float(level2_yawn_seconds)
        self.level2_combined_seconds = float(level2_combined_seconds)

        self.level3_eye_closed_seconds = float(level3_eye_closed_seconds)
        self.level3_no_face_seconds = float(level3_no_face_seconds)

        self.no_face_level1_seconds = float(no_face_level1_seconds)
        self.no_face_level2_seconds = float(no_face_level2_seconds)

        self.audio_cooldown = float(audio_cooldown)

        self.resize_width = int(resize_width)
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)

        self.min_detection_confidence = float(min_detection_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)

        # Buffer model
        self.feature_buffer = deque(maxlen=self.sequence_length)
        self.model_prob_buffer = deque(maxlen=max(1, self.model_smooth_window))

        # Timer rule
        self.eye_closed_start = None
        self.eye_low_start = None
        self.yawn_start = None
        self.no_face_start = None
        self.combined_risk_start = None

        # Head baseline
        self.pitch_baseline = deque(maxlen=90)

        # State
        self.model_drowsy_count = 0
        self.current_level = 0
        self.last_audio_time = 0.0
        self.last_rule_reason = "NORMAL"

        # Runtime objects
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_index = None
        self.output_index = None
        self.input_shape = None
        self.input_dtype = None
        self.input_scale = None
        self.input_zero_point = None

        self.scaler = None
        self.face_mesh = None

        self._load_tflite_model()
        self._load_scaler()
        self._init_audio()
        self._init_face_mesh()

    # ========================================================
    # INIT
    # ========================================================

    def _load_tflite_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy model: {self.model_path}")

        self.interpreter = Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_index = self.input_details[0]["index"]
        self.output_index = self.output_details[0]["index"]

        self.input_shape = self.input_details[0]["shape"]
        self.input_dtype = self.input_details[0]["dtype"]

        quant = self.input_details[0].get("quantization", (0.0, 0))
        self.input_scale = float(quant[0]) if quant else 0.0
        self.input_zero_point = int(quant[1]) if quant else 0

        print(f"[OK] Load model: {self.model_path}")
        print(f"[INFO] Input shape: {self.input_shape}")
        print(f"[INFO] Input dtype : {self.input_dtype}")
        print(f"[INFO] Output info : {self.output_details}")

        if len(self.input_shape) != 3:
            raise ValueError(f"Model cần input dạng [batch, sequence, features], hiện tại: {self.input_shape}")

        model_seq = int(self.input_shape[1])
        model_feat = int(self.input_shape[2])

        if model_seq > 0 and model_seq != self.sequence_length:
            print(f"[WARN] sequence_length code={self.sequence_length}, model={model_seq}. Tự cập nhật.")
            self.sequence_length = model_seq
            self.feature_buffer = deque(maxlen=self.sequence_length)

        if model_feat > 0 and model_feat != self.num_features:
            print(f"[WARN] num_features code={self.num_features}, model={model_feat}. Tự cập nhật.")
            self.num_features = model_feat

    def _load_scaler(self):
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Không tìm thấy scaler: {self.scaler_path}")

        self.scaler = joblib.load(self.scaler_path)
        print(f"[OK] Load scaler: {self.scaler_path}")

    def _init_audio(self):
        try:
            pygame.mixer.init()
            print("[OK] Pygame mixer ready")
        except Exception as e:
            print(f"[WARN] Không khởi tạo được âm thanh: {e}")

        for level, path in self.audio_paths.items():
            if not path.exists():
                print(f"[WARN] Không thấy âm thanh level {level}: {path}")

    def _init_face_mesh(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        print("[OK] MediaPipe FaceMesh ready")

    # ========================================================
    # FEATURE
    # ========================================================

    def resize_frame(self, frame):
        if self.resize_width <= 0:
            return frame

        h, w = frame.shape[:2]

        if w <= self.resize_width:
            return frame

        ratio = self.resize_width / float(w)
        new_h = int(h * ratio)

        return cv2.resize(frame, (self.resize_width, new_h))

    def extract_features(self, frame) -> Tuple[Optional[np.ndarray], np.ndarray]:
        frame = self.resize_frame(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None, frame

        landmarks = results.multi_face_landmarks[0].landmark
        features = extract_all_features(landmarks, frame.shape)

        if features is None:
            return None, frame

        if len(features) != self.num_features:
            print(f"[WARN] Sai số feature: {len(features)}, model cần {self.num_features}")
            return None, frame

        return features.astype(np.float32), frame

    # ========================================================
    # MODEL
    # ========================================================

    def add_features(self, features: np.ndarray):
        self.feature_buffer.append(features.astype(np.float32).tolist())

    def model_ready(self) -> bool:
        return len(self.feature_buffer) == self.sequence_length

    def _prepare_input_data(self, x_float32: np.ndarray) -> np.ndarray:
        """
        Hỗ trợ cả model float32 và model quantized.
        """
        if self.input_dtype == np.float32:
            return x_float32.astype(np.float32)

        if self.input_scale and self.input_scale > 0:
            x_quant = x_float32 / self.input_scale + self.input_zero_point
            x_quant = np.clip(x_quant, np.iinfo(self.input_dtype).min, np.iinfo(self.input_dtype).max)
            return x_quant.astype(self.input_dtype)

        return x_float32.astype(self.input_dtype)

    def predict_model(self) -> Tuple[float, int]:
        """
        Trả về:
            model_prob_smooth: xác suất drowsy đã làm mượt
            model_level: mức đề xuất riêng bởi model
        """
        if not self.model_ready():
            return 0.0, 0

        sequence = np.asarray(self.feature_buffer, dtype=np.float32)

        try:
            sequence_scaled = self.scaler.transform(sequence)
        except Exception as e:
            raise RuntimeError(
                "Scaler transform lỗi. Kiểm tra feature_scaler.joblib có đúng với 5 feature "
                "[ear_L, ear_R, mar, pitch, yaw] không."
            ) from e

        input_float = sequence_scaled.reshape(1, self.sequence_length, self.num_features).astype(np.float32)
        input_data = self._prepare_input_data(input_float)

        self.interpreter.set_tensor(self.input_index, input_data)
        self.interpreter.invoke()

        pred = np.asarray(self.interpreter.get_tensor(self.output_index))

        if pred.ndim == 2 and pred.shape[1] == 2:
            raw_prob = float(pred[0][1])
        elif pred.ndim == 2 and pred.shape[1] == 1:
            raw_prob = float(pred[0][0])
        else:
            raw_prob = float(pred.reshape(-1)[-1])

        raw_prob = float(np.clip(raw_prob, 0.0, 1.0))

        self.model_prob_buffer.append(raw_prob)
        prob = float(np.mean(self.model_prob_buffer))

        if prob >= self.model_threshold:
            self.model_drowsy_count += 1
        else:
            self.model_drowsy_count = max(0, self.model_drowsy_count - 1)

        # Rule riêng theo model cho 3 mức
        if prob >= 0.95 and self.model_drowsy_count >= self.model_consecutive_level3:
            model_level = 3
        elif prob >= 0.90 and self.model_drowsy_count >= self.model_consecutive_level2:
            model_level = 2
        elif prob >= self.model_threshold and self.model_drowsy_count >= self.model_consecutive_level1:
            model_level = 1
        else:
            model_level = 0

        return prob, model_level

    # ========================================================
    # RULE TIMERS
    # ========================================================

    def _timer(self, condition: bool, attr_name: str) -> float:
        now = time.time()

        if condition:
            if getattr(self, attr_name) is None:
                setattr(self, attr_name, now)
            return now - getattr(self, attr_name)

        setattr(self, attr_name, None)
        return 0.0

    def reset_drowsy_rules(self):
        self.eye_closed_start = None
        self.eye_low_start = None
        self.yawn_start = None
        self.combined_risk_start = None
        self.model_drowsy_count = 0
        self.current_level = 0

    # ========================================================
    # HYBRID RULES - 3 LEVEL KHÁC NHAU
    # ========================================================

    def evaluate_rules(
        self,
        features: Optional[np.ndarray],
        face_found: bool,
        model_prob: float,
        model_level: int,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Kết hợp model và rule.

        Level 1:
            - Model nghi ngủ gật nhẹ
            - Hoặc mắt lim dim ngắn
            - Hoặc mắt nhắm rất ngắn
            - Hoặc ngáp ngắn

        Level 2:
            - Mắt nhắm lâu hơn
            - Hoặc model chắc hơn
            - Hoặc kết hợp: model + mắt thấp / ngáp / cúi đầu
            - Hoặc mất mặt vừa lâu

        Level 3:
            - Mắt nhắm rất lâu
            - Hoặc mất mặt quá lâu
            - Hoặc model rất chắc + có dấu hiệu rule mạnh
        """
        now = time.time()

        info = {
            "avg_ear": 0.0,
            "mar": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "eye_low_duration": 0.0,
            "eye_closed_duration": 0.0,
            "yawn_duration": 0.0,
            "no_face_duration": 0.0,
            "head_drop": False,
            "model_prob": model_prob,
            "model_level": model_level,
            "model_count": self.model_drowsy_count,
            "reason": "NORMAL",
        }

        # ----------------------------------------------------
        # RULE MẤT MẶT
        # ----------------------------------------------------
        if not face_found or features is None:
            self.feature_buffer.clear()
            self.model_prob_buffer.clear()
            self.model_drowsy_count = 0

            if self.no_face_start is None:
                self.no_face_start = now

            no_face_duration = now - self.no_face_start
            info["no_face_duration"] = no_face_duration

            if no_face_duration >= self.level3_no_face_seconds:
                info["reason"] = "LEVEL 3: mat mat qua lau"
                return 3, info

            if no_face_duration >= self.no_face_level2_seconds:
                info["reason"] = "LEVEL 2: mat mat kha lau"
                return 2, info

            if no_face_duration >= self.no_face_level1_seconds:
                info["reason"] = "LEVEL 1: khong thay mat"
                return 1, info

            info["reason"] = "NORMAL: no face short"
            return 0, info

        self.no_face_start = None

        # ----------------------------------------------------
        # LẤY FEATURE
        # ----------------------------------------------------
        ear_l, ear_r, mar, pitch, yaw = [float(x) for x in features]
        avg_ear = (ear_l + ear_r) / 2.0

        info["avg_ear"] = avg_ear
        info["mar"] = mar
        info["pitch"] = pitch
        info["yaw"] = yaw

        eye_low = avg_ear < self.ear_low
        eye_closed = avg_ear < self.ear_closed
        yawn = mar > self.mar_yawn

        eye_low_duration = self._timer(eye_low, "eye_low_start")
        eye_closed_duration = self._timer(eye_closed, "eye_closed_start")
        yawn_duration = self._timer(yawn, "yawn_start")

        info["eye_low_duration"] = eye_low_duration
        info["eye_closed_duration"] = eye_closed_duration
        info["yawn_duration"] = yawn_duration

        # ----------------------------------------------------
        # HEAD DROP
        # ----------------------------------------------------
        # Cập nhật baseline khi có vẻ bình thường
        if avg_ear > self.ear_low and mar < self.mar_yawn and model_prob < 0.55:
            self.pitch_baseline.append(pitch)

        if len(self.pitch_baseline) >= 15:
            pitch_base = float(np.median(self.pitch_baseline))
        else:
            pitch_base = pitch

        head_drop = abs(pitch - pitch_base) > self.pitch_delta
        info["head_drop"] = bool(head_drop)

        # ----------------------------------------------------
        # COMBINED RISK
        # ----------------------------------------------------
        combined_risk = False

        # Model hơi cao + mắt thấp
        if model_prob >= 0.70 and eye_low:
            combined_risk = True

        # Model hơi cao + ngáp
        if model_prob >= 0.65 and yawn:
            combined_risk = True

        # Mắt thấp + cúi/gật đầu
        if eye_low and head_drop:
            combined_risk = True

        combined_duration = self._timer(combined_risk, "combined_risk_start")

        # ====================================================
        # LEVEL 3 - KHẨN CẤP
        # ====================================================

        # Mắt nhắm rất lâu
        if eye_closed_duration >= self.level3_eye_closed_seconds:
            info["reason"] = "LEVEL 3: mat nham rat lau"
            return 3, info

        # Model rất chắc + mắt nhắm tương đối lâu
        if model_prob >= 0.95 and eye_closed_duration >= 2.5:
            info["reason"] = "LEVEL 3: model rat chac + mat nham"
            return 3, info

        # Model rất chắc + nhiều lần liên tiếp
        if model_level >= 3:
            info["reason"] = "LEVEL 3: model bao drowsy lien tiep rat cao"
            return 3, info

        # Mắt thấp + cúi đầu kéo dài
        if eye_low_duration >= 3.0 and head_drop:
            info["reason"] = "LEVEL 3: mat lim dim + cui/gat dau"
            return 3, info

        # ====================================================
        # LEVEL 2 - NGUY HIỂM
        # ====================================================

        # Mắt nhắm lâu
        if eye_closed_duration >= self.level2_eye_closed_seconds:
            info["reason"] = "LEVEL 2: mat nham lau"
            return 2, info

        # Ngáp lâu + model có nghi ngờ
        if yawn_duration >= self.level2_yawn_seconds and model_prob >= 0.60:
            info["reason"] = "LEVEL 2: ngap lau + model nghi ngo"
            return 2, info

        # Model mức 2
        if model_level >= 2:
            info["reason"] = "LEVEL 2: model bao drowsy on dinh"
            return 2, info

        # Kết hợp nhiều dấu hiệu kéo dài
        if combined_duration >= self.level2_combined_seconds:
            info["reason"] = "LEVEL 2: ket hop model + rule"
            return 2, info

        # ====================================================
        # LEVEL 1 - NHẮC NHỞ
        # ====================================================

        # Mắt lim dim ngắn
        if eye_low_duration >= self.level1_eye_low_seconds:
            info["reason"] = "LEVEL 1: mat lim dim"
            return 1, info

        # Mắt nhắm ngắn
        if eye_closed_duration >= self.level1_eye_closed_seconds:
            info["reason"] = "LEVEL 1: mat nham ngan"
            return 1, info

        # Ngáp ngắn
        if yawn_duration >= self.level1_yawn_seconds:
            info["reason"] = "LEVEL 1: ngap"
            return 1, info

        # Model mức 1
        if model_level >= 1:
            info["reason"] = "LEVEL 1: model nghi ngo"
            return 1, info

        info["reason"] = "NORMAL"
        return 0, info

    # ========================================================
    # AUDIO
    # ========================================================

    def play_alert(self, level: int):
        if level <= 0:
            return

        now = time.time()

        if now - self.last_audio_time < self.audio_cooldown:
            return

        path = self.audio_paths.get(level)

        if path is None or not path.exists():
            return

        try:
            pygame.mixer.music.load(str(path))
            pygame.mixer.music.play()
            self.last_audio_time = now
        except Exception as e:
            print(f"[WARN] Không phát được âm thanh level {level}: {e}")

    # ========================================================
    # DISPLAY
    # ========================================================

    def draw_status(self, frame, level: int, info: Dict[str, Any], face_found: bool):
        status = "NORMAL" if level == 0 else f"DROWSY - LEVEL {level}"

        if level == 0:
            color = (0, 255, 0)
        elif level == 1:
            color = (0, 255, 255)
        elif level == 2:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)

        y = 35
        line_h = 32

        def put(text, yy, col=(255, 255, 255), scale=0.65, thick=2):
            cv2.putText(
                frame,
                text,
                (20, yy),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                col,
                thick,
            )

        put(f"Status: {status}", y, color, 0.85, 2)
        y += line_h

        put(f"Reason: {info.get('reason', '')}", y, color, 0.62, 2)
        y += line_h

        put(
            f"Model prob: {info.get('model_prob', 0):.2f} | "
            f"Model level: {info.get('model_level', 0)} | "
            f"Count: {info.get('model_count', 0)}",
            y,
        )
        y += line_h

        put(
            f"EAR: {info.get('avg_ear', 0):.3f} | "
            f"MAR: {info.get('mar', 0):.3f} | "
            f"Pitch: {info.get('pitch', 0):.3f}",
            y,
        )
        y += line_h

        put(
            f"Eye low: {info.get('eye_low_duration', 0):.1f}s | "
            f"Eye closed: {info.get('eye_closed_duration', 0):.1f}s | "
            f"Yawn: {info.get('yawn_duration', 0):.1f}s",
            y,
        )
        y += line_h

        put(
            f"Face: {'FOUND' if face_found else 'NOT FOUND'} | "
            f"No face: {info.get('no_face_duration', 0):.1f}s | "
            f"Buffer: {len(self.feature_buffer)}/{self.sequence_length}",
            y,
            (0, 255, 0) if face_found else (0, 0, 255),
        )
        y += line_h

        put(
            f"Head drop: {info.get('head_drop', False)}",
            y,
        )

        return frame

    # ========================================================
    # PROCESS FRAME
    # ========================================================

    def process_frame(self, frame):
        features, frame = self.extract_features(frame)
        face_found = features is not None

        model_prob = 0.0
        model_level = 0

        if face_found:
            self.add_features(features)

            if self.model_ready():
                model_prob, model_level = self.predict_model()
        else:
            # Không thấy mặt thì không dùng chuỗi feature cũ nữa
            self.feature_buffer.clear()
            self.model_prob_buffer.clear()

        level, info = self.evaluate_rules(
            features=features,
            face_found=face_found,
            model_prob=model_prob,
            model_level=model_level,
        )

        self.current_level = level
        self.last_rule_reason = info.get("reason", "")

        self.play_alert(level)

        frame = self.draw_status(
            frame=frame,
            level=level,
            info=info,
            face_found=face_found,
        )

        result = {
            "level": level,
            "face_found": face_found,
            "model_prob": model_prob,
            "model_level": model_level,
            "rule_info": info,
        }

        return frame, result

    # ========================================================
    # CAMERA
    # ========================================================

    def run_camera(self, camera_id: int = 0):
        if os.name == "nt":
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(camera_id)

        if self.camera_width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)

        if self.camera_height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

        if not cap.isOpened():
            raise RuntimeError(f"Không mở được camera: {camera_id}")

        print("[INFO] Camera running...")
        print("[INFO] Press Q to quit.")

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("[WARN] Không đọc được frame từ camera.")
                    break

                output_frame, result = self.process_frame(frame)

                cv2.imshow("Drowsiness Detector Runtime", output_frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.close()

    def close(self):
        try:
            if self.face_mesh is not None:
                self.face_mesh.close()
        except Exception:
            pass

        try:
            pygame.mixer.quit()
        except Exception:
            pass


# ============================================================
# ARGPARSE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Drowsiness Detector Runtime - TFLite + Rules + 3 Alert Levels"
    )

    parser.add_argument("--camera", type=int, default=0)

    parser.add_argument(
        "--model",
        default=str(PROJECT_ROOT / "models" / "drowsiness_gru.tflite"),
    )

    parser.add_argument(
        "--scaler",
        default=str(PROJECT_ROOT / "models" / "feature_scaler.joblib"),
    )

    parser.add_argument(
        "--audio1",
        default=str(PROJECT_ROOT / "audios" / "alert_lv1.wav"),
    )

    parser.add_argument(
        "--audio2",
        default=str(PROJECT_ROOT / "audios" / "alert_lv2.wav"),
    )

    parser.add_argument(
        "--audio3",
        default=str(PROJECT_ROOT / "audios" / "alert_lv3.wav"),
    )

    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--num-features", type=int, default=5)

    parser.add_argument("--model-threshold", type=float, default=0.85)
    parser.add_argument("--model-smooth-window", type=int, default=10)

    parser.add_argument("--ear-closed", type=float, default=0.20)
    parser.add_argument("--ear-low", type=float, default=0.24)
    parser.add_argument("--mar-yawn", type=float, default=0.65)
    parser.add_argument("--pitch-delta", type=float, default=0.18)

    parser.add_argument("--audio-cooldown", type=float, default=3.0)

    parser.add_argument("--resize-width", type=int, default=640)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)

    return parser.parse_args()


def main():
    args = parse_args()

    detector = DrowsinessDetectorTFLiteRuntime(
        model_path=args.model,
        scaler_path=args.scaler,
        audio_level1_path=args.audio1,
        audio_level2_path=args.audio2,
        audio_level3_path=args.audio3,

        sequence_length=args.sequence_length,
        num_features=args.num_features,

        model_threshold=args.model_threshold,
        model_smooth_window=args.model_smooth_window,

        ear_closed=args.ear_closed,
        ear_low=args.ear_low,
        mar_yawn=args.mar_yawn,
        pitch_delta=args.pitch_delta,

        audio_cooldown=args.audio_cooldown,

        resize_width=args.resize_width,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
    )

    detector.run_camera(camera_id=args.camera)


if __name__ == "__main__":
    main()