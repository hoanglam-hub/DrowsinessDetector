"""
Extract MediaPipe FaceMesh features into sequence CSV files for training.

Input folder:
    data/normal/*.mp4
    data/drowsy/*.mp4

Output:
    data/processed/normal.csv
    data/processed/drowsy.csv

Each row is one sequence of 30 frames.
Feature order per frame:
    ear_L, ear_R, mar, pitch, yaw
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_utils import extract_all_features

try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
except AttributeError:
    from mediapipe.python.solutions import face_mesh as mp_face_mesh


CLASSES = ["normal", "drowsy"]
FEATURE_NAMES = ["ear_L", "ear_R", "mar", "pitch", "yaw"]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def create_csv_header(sequence_length: int) -> list[str]:
    header = ["video_name", "class_name", "sequence_id"]
    for frame_idx in range(1, sequence_length + 1):
        for name in FEATURE_NAMES:
            header.append(f"frame_{frame_idx}_{name}")
    header.append("label")
    return header


def iter_video_files(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        return []
    return (p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def resize_keep_ratio(frame, resize_width: int | None):
    if not resize_width or resize_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= resize_width:
        return frame
    ratio = resize_width / float(w)
    return cv2.resize(frame, (resize_width, int(h * ratio)))


def process_video(
    video_path: Path,
    class_name: str,
    label_idx: int,
    writer: csv.writer,
    face_mesh,
    sequence_length: int,
    stride: int,
    resize_width: int | None,
    reset_on_missing_face: bool,
    frame_skip: int,
) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open video: {video_path}")
        return 0, 0, 0

    buffer: list[list[float]] = []
    sequence_id = 0
    total_frames = 0
    detected_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        total_frames += 1
        if frame_skip > 1 and (total_frames % frame_skip != 0):
            continue

        frame = resize_keep_ratio(frame, resize_width)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            if reset_on_missing_face:
                buffer.clear()
            continue

        detected_frames += 1
        landmarks = results.multi_face_landmarks[0].landmark
        features = extract_all_features(landmarks, frame.shape)

        if features is None or len(features) != len(FEATURE_NAMES):
            if reset_on_missing_face:
                buffer.clear()
            continue

        arr = np.asarray(features, dtype=np.float32)
        if not np.all(np.isfinite(arr)):
            if reset_on_missing_face:
                buffer.clear()
            continue

        buffer.append([float(x) for x in arr])

        if len(buffer) == sequence_length:
            row = [video_path.name, class_name, sequence_id]
            for frame_features in buffer:
                row.extend(frame_features)
            row.append(label_idx)
            writer.writerow(row)
            sequence_id += 1

            if stride >= sequence_length:
                buffer.clear()
            else:
                buffer = buffer[stride:]

    cap.release()
    return sequence_id, total_frames, detected_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--stride", type=int, default=10, help="10 = nhiều mẫu hơn, 30 = không chồng lấn")
    parser.add_argument("--resize-width", type=int, default=640)
    parser.add_argument("--frame-skip", type=int, default=1, help="1 = lấy mọi frame, 2 = lấy cách 1 frame")
    parser.add_argument("--keep-through-missing-face", action="store_true")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reset_on_missing_face = not args.keep_through_missing_face
    resize_width = None if args.resize_width <= 0 else args.resize_width

    print("=" * 70)
    print("EXTRACT FACE FEATURES")
    print("=" * 70)
    print(f"Data dir        : {data_dir}")
    print(f"Output dir      : {out_dir}")
    print(f"Sequence length : {args.sequence_length}")
    print(f"Stride          : {args.stride}")
    print(f"Resize width    : {resize_width}")
    print(f"Frame skip      : {args.frame_skip}")
    print(f"Reset missing   : {reset_on_missing_face}")

    total_sequences = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as face_mesh:
        for label_idx, class_name in enumerate(CLASSES):
            input_folder = data_dir / class_name
            output_csv = out_dir / f"{class_name}.csv"
            videos = list(iter_video_files(input_folder))

            if not videos:
                print(f"\n[WARN] No videos found in: {input_folder}")
                continue

            print(f"\n[{class_name}] {len(videos)} videos -> {output_csv}")
            class_sequences = 0
            class_frames = 0
            class_detected = 0

            with output_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(create_csv_header(args.sequence_length))

                for video_path in videos:
                    seq_count, frame_count, detected_count = process_video(
                        video_path=video_path,
                        class_name=class_name,
                        label_idx=label_idx,
                        writer=writer,
                        face_mesh=face_mesh,
                        sequence_length=args.sequence_length,
                        stride=args.stride,
                        resize_width=resize_width,
                        reset_on_missing_face=reset_on_missing_face,
                        frame_skip=max(1, args.frame_skip),
                    )
                    class_sequences += seq_count
                    class_frames += frame_count
                    class_detected += detected_count
                    rate = detected_count / frame_count * 100 if frame_count else 0.0
                    print(f"  + {video_path.name}: {seq_count} seq | face {detected_count}/{frame_count} ({rate:.1f}%)")

            total_sequences += class_sequences
            print(f"  => {class_name}: {class_sequences} sequences")

    print("\nDONE")
    print(f"Total sequences: {total_sequences}")
    print(f"CSV saved to   : {out_dir}")


if __name__ == "__main__":
    main()
