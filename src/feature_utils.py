"""
Feature utilities for drowsiness detection.
Feature order is fixed:
    [ear_L, ear_R, mar, pitch, yaw]

Keep this file identical for extraction, training, and realtime inference.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import cv2
import numpy as np
from scipy.spatial import distance as dist


LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [78, 81, 13, 311, 308, 402, 14, 178]

# 3D model points for solvePnP, approximate generic face model.
# Used only to derive pitch/yaw more stably than simple ratios.
POSE_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
}
MODEL_POINTS_3D = np.array([
    [0.0, 0.0, 0.0],        # nose tip
    [0.0, -63.6, -12.5],    # chin
    [-43.3, 32.7, -26.0],   # left eye outer
    [43.3, 32.7, -26.0],    # right eye outer
    [-28.9, -28.9, -24.1],  # mouth left
    [28.9, -28.9, -24.1],   # mouth right
], dtype=np.float64)


def _point(landmarks, index: int, width: int, height: int) -> tuple[float, float]:
    lm = landmarks[index]
    return float(lm.x * width), float(lm.y * height)


def calculate_ear(eye_points: Sequence[Sequence[float]]) -> float:
    eye = np.asarray(eye_points, dtype=np.float32)
    if eye.shape != (6, 2):
        return float("nan")
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])
    if c <= 1e-6:
        return float("nan")
    return float((a + b) / (2.0 * c))


def calculate_mar(mouth_points: Sequence[Sequence[float]]) -> float:
    mouth = np.asarray(mouth_points, dtype=np.float32)
    if mouth.shape != (8, 2):
        return float("nan")
    a = dist.euclidean(mouth[1], mouth[7])
    b = dist.euclidean(mouth[2], mouth[6])
    c = dist.euclidean(mouth[3], mouth[5])
    d = dist.euclidean(mouth[0], mouth[4])
    if d <= 1e-6:
        return float("nan")
    return float((a + b + c) / (2.0 * d))


def _rotation_matrix_to_euler_angles(rotation_matrix: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0.0
    return math.degrees(x), math.degrees(y), math.degrees(z)


def get_head_pose(landmarks, frame_shape) -> tuple[float, float]:
    """Return normalized pitch and yaw in roughly degree/45 scale."""
    height, width = frame_shape[:2]
    image_points = np.array([
        _point(landmarks, POSE_LANDMARKS["nose_tip"], width, height),
        _point(landmarks, POSE_LANDMARKS["chin"], width, height),
        _point(landmarks, POSE_LANDMARKS["left_eye_outer"], width, height),
        _point(landmarks, POSE_LANDMARKS["right_eye_outer"], width, height),
        _point(landmarks, POSE_LANDMARKS["mouth_left"], width, height),
        _point(landmarks, POSE_LANDMARKS["mouth_right"], width, height),
    ], dtype=np.float64)

    focal_length = float(width)
    center = (width / 2.0, height / 2.0)
    camera_matrix = np.array([
        [focal_length, 0.0, center[0]],
        [0.0, focal_length, center[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    try:
        ok, rotation_vector, translation_vector = cv2.solvePnP(
            MODEL_POINTS_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return 0.0, 0.0
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pitch_deg, yaw_deg, _ = _rotation_matrix_to_euler_angles(rotation_matrix)
        # Normalize so the scaler sees stable values, not huge raw degrees.
        pitch = float(np.clip(pitch_deg / 45.0, -2.0, 2.0))
        yaw = float(np.clip(yaw_deg / 45.0, -2.0, 2.0))
        return pitch, yaw
    except Exception:
        return 0.0, 0.0


def extract_all_features(landmarks, frame_shape) -> list[float] | None:
    height, width = frame_shape[:2]

    left_eye = [_point(landmarks, i, width, height) for i in LEFT_EYE_IDX]
    right_eye = [_point(landmarks, i, width, height) for i in RIGHT_EYE_IDX]
    mouth = [_point(landmarks, i, width, height) for i in MOUTH_IDX]

    ear_left = calculate_ear(left_eye)
    ear_right = calculate_ear(right_eye)
    mar = calculate_mar(mouth)
    pitch, yaw = get_head_pose(landmarks, frame_shape)

    features = np.asarray([ear_left, ear_right, mar, pitch, yaw], dtype=np.float32)
    if not np.all(np.isfinite(features)):
        return None
    return features.tolist()
