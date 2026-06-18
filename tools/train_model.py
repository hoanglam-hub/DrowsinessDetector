"""
Train a drowsiness sequence model from CSV files created by tools/extract_to_csv.py.

Outputs:
    models/drowsiness_gru.keras
    models/drowsiness_gru.tflite
    models/feature_scaler.joblib
    models/class_names.json
    models/feature_columns.json
    models/training_history.csv
    models/classification_report.txt
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, GRU, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["normal", "drowsy"]
LABEL_COLUMN = "label"
METADATA_COLUMNS = {"video_name", "class_name", "sequence_id", LABEL_COLUMN}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_feature_columns(df: pd.DataFrame, sequence_length: int, num_features: int) -> list[str]:
    feature_cols = [c for c in df.columns if c not in METADATA_COLUMNS]
    expected = sequence_length * num_features
    if len(feature_cols) != expected:
        raise ValueError(
            f"Wrong feature count: got {len(feature_cols)}, expected {expected} "
            f"= {sequence_length} frames x {num_features} features."
        )
    return feature_cols


def load_dataset(data_dir: Path, sequence_length: int, num_features: int) -> tuple[pd.DataFrame, list[str]]:
    csv_files = sorted(glob.glob(str(data_dir / "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}. Run tools/extract_to_csv.py first.")

    dfs = []
    for file in csv_files:
        print(f"  - Reading: {file}")
        df = pd.read_csv(file)
        if LABEL_COLUMN not in df.columns:
            raise ValueError(f"CSV missing label column: {file}")
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.dropna().reset_index(drop=True)
    feature_cols = get_feature_columns(df_all, sequence_length, num_features)

    values = df_all[feature_cols].values.astype(np.float32)
    valid_mask = np.all(np.isfinite(values), axis=1)
    if not np.all(valid_mask):
        print(f"[WARN] Removing invalid rows: {int(np.sum(~valid_mask))}")
        df_all = df_all.loc[valid_mask].reset_index(drop=True)

    return df_all, feature_cols


def split_by_video_if_possible(df: pd.DataFrame, seed: int, val_ratio: float, test_ratio: float):
    y = df[LABEL_COLUMN].values.astype(int)

    if "video_name" not in df.columns:
        idx = np.arange(len(df))
        train_idx, temp_idx = train_test_split(idx, test_size=val_ratio + test_ratio, random_state=seed, stratify=y)
        temp_y = y[temp_idx]
        rel_test = test_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(temp_idx, test_size=rel_test, random_state=seed, stratify=temp_y)
        return train_idx, val_idx, test_idx

    enough_videos = True
    for label in sorted(df[LABEL_COLUMN].unique()):
        n_videos = df.loc[df[LABEL_COLUMN] == label, "video_name"].nunique()
        if n_videos < 3:
            enough_videos = False
            print(f"[WARN] Class {label} has only {n_videos} video(s). Falling back to sequence split.")

    if not enough_videos:
        idx = np.arange(len(df))
        train_idx, temp_idx = train_test_split(idx, test_size=val_ratio + test_ratio, random_state=seed, stratify=y)
        temp_y = y[temp_idx]
        rel_test = test_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(temp_idx, test_size=rel_test, random_state=seed, stratify=temp_y)
        return train_idx, val_idx, test_idx

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for label in sorted(df[LABEL_COLUMN].unique()):
        class_df = df[df[LABEL_COLUMN] == label]
        videos = class_df["video_name"].drop_duplicates().to_numpy()
        rng.shuffle(videos)

        n = len(videos)
        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_test + n_val >= n:
            n_test = 1
            n_val = 1

        test_videos = set(videos[:n_test])
        val_videos = set(videos[n_test:n_test + n_val])
        train_videos = set(videos[n_test + n_val:])

        train_indices.extend(class_df[class_df["video_name"].isin(train_videos)].index.tolist())
        val_indices.extend(class_df[class_df["video_name"].isin(val_videos)].index.tolist())
        test_indices.extend(class_df[class_df["video_name"].isin(test_videos)].index.tolist())

    return np.array(train_indices), np.array(val_indices), np.array(test_indices)


def build_model(sequence_length: int, num_features: int, num_classes: int) -> Sequential:
    model = Sequential([
        Input(shape=(sequence_length, num_features)),
        GRU(64, return_sequences=True, unroll=True, reset_after=True),
        Dropout(0.25),
        GRU(32, return_sequences=False, unroll=True, reset_after=True),
        Dropout(0.25),
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dropout(0.20),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_xy(df: pd.DataFrame, indices: Iterable[int], feature_cols: list[str], sequence_length: int, num_features: int):
    sub = df.iloc[list(indices)]
    x_flat = sub[feature_cols].values.astype(np.float32)
    y_raw = sub[LABEL_COLUMN].values.astype(int)
    x = x_flat.reshape(-1, sequence_length, num_features)
    return x, y_raw


def convert_to_tflite(model, tflite_path: Path, select_ops_path: Path) -> None:
    print("\nConverting to TFLite built-in...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # Do not quantize here; keep maximum compatibility first.
        tflite_model = converter.convert()
        tflite_path.write_bytes(tflite_model)
        print(f"[OK] TFLite saved: {tflite_path}")
        return
    except Exception as e:
        print(f"[WARN] Built-in TFLite conversion failed: {e}")

    print("Trying SELECT_TF_OPS conversion...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter._experimental_lower_tensor_list_ops = False
        tflite_model = converter.convert()
        select_ops_path.write_bytes(tflite_model)
        print(f"[OK] SELECT_OPS TFLite saved: {select_ops_path}")
    except Exception as e:
        print(f"[ERROR] SELECT_OPS conversion also failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--models-dir", default=str(ROOT / "models"))
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--num-features", type=int, default=5)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "drowsiness_gru.keras"
    tflite_path = models_dir / "drowsiness_gru.tflite"
    select_ops_path = models_dir / "drowsiness_gru_select_ops.tflite"
    scaler_path = models_dir / "feature_scaler.joblib"
    class_names_path = models_dir / "class_names.json"
    feature_columns_path = models_dir / "feature_columns.json"
    history_path = models_dir / "training_history.csv"
    report_path = models_dir / "classification_report.txt"

    print("=" * 70)
    print("TRAIN DROWSINESS MODEL")
    print("=" * 70)

    df, feature_cols = load_dataset(data_dir, args.sequence_length, args.num_features)
    print(f"Total sequences: {len(df)}")

    y_all = df[LABEL_COLUMN].values.astype(int)
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i} - {name}: {int(np.sum(y_all == i))}")

    train_idx, val_idx, test_idx = split_by_video_if_possible(df, args.seed, args.val_ratio, args.test_ratio)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    x_train, y_train_raw = make_xy(df, train_idx, feature_cols, args.sequence_length, args.num_features)
    x_val, y_val_raw = make_xy(df, val_idx, feature_cols, args.sequence_length, args.num_features)
    x_test, y_test_raw = make_xy(df, test_idx, feature_cols, args.sequence_length, args.num_features)

    print("\nSplit:")
    print(f"  Train: {x_train.shape[0]}")
    print(f"  Val  : {x_val.shape[0]}")
    print(f"  Test : {x_test.shape[0]}")

    scaler = StandardScaler()
    scaler.fit(x_train.reshape(-1, args.num_features))

    def scale_x(x: np.ndarray) -> np.ndarray:
        n, s, f = x.shape
        scaled = scaler.transform(x.reshape(-1, f))
        return scaled.reshape(n, s, f).astype(np.float32)

    x_train = scale_x(x_train)
    x_val = scale_x(x_val)
    x_test = scale_x(x_test)

    joblib.dump(scaler, scaler_path)
    class_names_path.write_text(json.dumps(CLASS_NAMES, ensure_ascii=False, indent=2), encoding="utf-8")
    feature_columns_path.write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Scaler saved: {scaler_path}")

    y_train = to_categorical(y_train_raw, num_classes=args.num_classes)
    y_val = to_categorical(y_val_raw, num_classes=args.num_classes)
    y_test = to_categorical(y_test_raw, num_classes=args.num_classes)

    class_weight_values = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(args.num_classes),
        y=y_train_raw,
    )
    class_weights = {i: float(w) for i, w in enumerate(class_weight_values)}
    print(f"Class weights: {class_weights}")

    model = build_model(args.sequence_length, args.num_features, args.num_classes)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=18, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-5, verbose=1),
        ModelCheckpoint(filepath=str(model_path), monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    history = model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    pd.DataFrame(history.history).to_csv(history_path, index=False)

    best_model = load_model(model_path)
    print("\nEvaluating best model...")
    test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=1)
    pred_prob = best_model.predict(x_test, verbose=0)
    pred = np.argmax(pred_prob, axis=1)

    cm = confusion_matrix(y_test_raw, pred)
    report = classification_report(y_test_raw, pred, target_names=CLASS_NAMES, digits=4, zero_division=0)

    print("\n" + "=" * 70)
    print("TEST RESULT")
    print("=" * 70)
    print(f"Test loss: {test_loss:.6f}")
    print(f"Test acc : {test_acc:.6f}")
    print("Confusion matrix:")
    print(cm)
    print(report)

    report_path.write_text(
        f"Test loss: {test_loss:.6f}\nTest accuracy: {test_acc:.6f}\n\nConfusion matrix:\n{cm}\n\n{report}\n",
        encoding="utf-8",
    )

    convert_to_tflite(best_model, tflite_path, select_ops_path)

    print("\nDONE")
    print(f"Keras model : {model_path}")
    print(f"TFLite model: {tflite_path}")
    print(f"Scaler      : {scaler_path}")


if __name__ == "__main__":
    main()
