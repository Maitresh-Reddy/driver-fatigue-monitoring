#!/usr/bin/env python3
"""
Complete setup, data preparation, and model training script.

This script:
1. Downloads datasets from Kaggle
2. Organizes data into correct structure
3. Trains CNN models
4. Evaluates performance
5. Tests the full system
"""

import os
import sys
import shutil
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))

from src.training import DataLoader, EyeStateModel, YawDetectionModel
from src.config import (
    CNN_INPUT_SIZE, CNN_BATCH_SIZE, CNN_EPOCHS, CNN_LEARNING_RATE,
    EYE_DATASET_PATH, YAWN_DATASET_PATH, EVALUATION_VIDEO_PATH,
    EYE_MODEL_PATH, YAWN_MODEL_PATH, DISTRACTION_MODEL_PATH, DROWSINESS_MODEL_PATH, RESULTS_DIR
)


class DatasetManager:
    """Manages dataset download and organization."""

    def __init__(self):
        self.dataset_dir = Path(__file__).parent / "dataset"
        self.raw_dir = self.dataset_dir / "raw"

    @staticmethod
    def _iter_files(root: Path, patterns):
        files = []
        for pattern in patterns:
            files.extend(root.rglob(pattern))
        return files

    @staticmethod
    def _count_images(root: Path):
        if not root.exists():
            return 0
        patterns = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
        total = 0
        for pattern in patterns:
            total += len(list(root.glob(pattern)))
        return total

    @staticmethod
    def _copy_if_missing(src: Path, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy(str(src), str(dst))

    @staticmethod
    def _normalize_col_name(name: str) -> str:
        return str(name).strip().lower().replace(' ', '')

    @staticmethod
    def _label_to_int(value):
        try:
            return int(float(value))
        except Exception:
            return 0

    @staticmethod
    def _resolve_image_path(dataset_root: Path, split_dir: Path, filename: str):
        candidate_paths = [
            split_dir / filename,
            split_dir / 'images' / filename,
            dataset_root / filename,
            dataset_root / 'images' / filename,
        ]
        for path in candidate_paths:
            if path.exists():
                return path

        matches = list(dataset_root.rglob(filename))
        if matches:
            return matches[0]
        return None

    @staticmethod
    def _extract_frames_from_video(video_path: Path, output_dir: Path, prefix: str,
                                   frame_step: int = 30, max_frames: int = 80):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0

        frame_index = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_step == 0:
                output_path = output_dir / f"{prefix}_{frame_index:06d}.jpg"
                if not output_path.exists():
                    cv2.imwrite(str(output_path), frame)
                    saved_count += 1
                if saved_count >= max_frames:
                    break

            frame_index += 1

        cap.release()
        return saved_count

    def download_datasets(self):
        """Download all datasets from Kaggle."""
        print("\n" + "=" * 70)
        print("DOWNLOADING DATASETS FROM KAGGLE")
        print("=" * 70)

        try:
            import kagglehub
        except ImportError:
            print("\n⚠️  kagglehub not installed. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
            import kagglehub

        datasets = {
            'cew': {'name': 'ahamedfarouk/cew-dataset', 'output': 'cew'},
            'ntu': {'name': 'banudeep/nthuddd2', 'output': 'ntu'},
            'yawn_kaggle': {'name': 'davidvazquezcic/yawn-dataset', 'output': 'yawn_kaggle'},
            'driver_distraction': {'name': 'n1ghtf4l1/driver-distraction', 'output': 'driver_distraction'},
            'drowsy_driver': {'name': 'abdurrahimcs50/drowsy-driver', 'output': 'drowsy_driver'},
        }

        self.raw_dir.mkdir(parents=True, exist_ok=True)

        for key, dataset_info in datasets.items():
            print(f"\n📥 Downloading {key.upper()} Dataset: {dataset_info['name']}")
            try:
                path = kagglehub.dataset_download(dataset_info['name'])
                output_path = self.raw_dir / dataset_info['output']
                if Path(path).exists() and Path(path) != output_path:
                    shutil.copytree(path, output_path, dirs_exist_ok=True)
                    print(f"✓ {key.upper()} Dataset downloaded to {output_path}")
                else:
                    print(f"✓ Dataset already at {path}")
            except Exception as e:
                print(f"✗ Error downloading {key} dataset: {e}")

    def _import_labeled_dataset_to_eye_yawn(self, dataset_root: Path, dataset_name: str,
                                            eye_open_dir: Path, eye_closed_dir: Path,
                                            yawn_dir: Path, non_yawn_dir: Path):
        """Import additional labeled samples using _classes.csv files."""
        if not dataset_root.exists():
            return {
                'eye_open': 0,
                'eye_closed': 0,
                'yawn': 0,
                'non_yawn': 0,
            }

        added = {
            'eye_open': 0,
            'eye_closed': 0,
            'yawn': 0,
            'non_yawn': 0,
        }

        csv_files = list(dataset_root.rglob('_classes.csv'))
        if not csv_files:
            return added

        for csv_path in csv_files:
            split_dir = csv_path.parent
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue

            if 'filename' not in df.columns:
                continue

            normalized_columns = {self._normalize_col_name(col): col for col in df.columns}
            col_d0 = normalized_columns.get('d0-eyesclosed')
            col_d1 = normalized_columns.get('d1-yawning')
            col_d3 = normalized_columns.get('d3-eyesopen')
            col_yawning = normalized_columns.get('yawning')

            for _, row in df.iterrows():
                filename = str(row.get('filename', '')).strip()
                if not filename:
                    continue

                src = self._resolve_image_path(dataset_root, split_dir, filename)
                if src is None:
                    continue

                prefix = f"{dataset_name}_{split_dir.name}"

                # Eye labels
                if col_d0 is not None and self._label_to_int(row.get(col_d0, 0)) == 1:
                    dst = eye_closed_dir / f"{prefix}_{filename}"
                    self._copy_if_missing(src, dst)
                    added['eye_closed'] += 1
                if col_d3 is not None and self._label_to_int(row.get(col_d3, 0)) == 1:
                    dst = eye_open_dir / f"{prefix}_{filename}"
                    self._copy_if_missing(src, dst)
                    added['eye_open'] += 1

                # Yawn labels
                yawn_positive = False
                if col_d1 is not None and self._label_to_int(row.get(col_d1, 0)) == 1:
                    yawn_positive = True
                if col_yawning is not None and self._label_to_int(row.get(col_yawning, 0)) == 1:
                    yawn_positive = True

                if yawn_positive:
                    dst = yawn_dir / f"{prefix}_{filename}"
                    self._copy_if_missing(src, dst)
                    added['yawn'] += 1
                elif col_d1 is not None or col_yawning is not None:
                    dst = non_yawn_dir / f"{prefix}_{filename}"
                    self._copy_if_missing(src, dst)
                    added['non_yawn'] += 1

        return added

    def organize_eye_dataset(self):
        """Organize CEW dataset into open/closed structure."""
        print("\n" + "=" * 70)
        print("ORGANIZING EYE DATASET")
        print("=" * 70)

        cew_path = self.raw_dir / 'cew'
        if not cew_path.exists():
            print(f"⚠️  CEW dataset not found at {cew_path}")
            print("Skipping eye dataset organization")
            return False

        eye_open_dir = EYE_DATASET_PATH / 'open'
        eye_closed_dir = EYE_DATASET_PATH / 'closed'
        yawn_dir = YAWN_DATASET_PATH / 'yawn'
        non_yawn_dir = YAWN_DATASET_PATH / 'non_yawn'

        eye_open_dir.mkdir(parents=True, exist_ok=True)
        eye_closed_dir.mkdir(parents=True, exist_ok=True)
        yawn_dir.mkdir(parents=True, exist_ok=True)
        non_yawn_dir.mkdir(parents=True, exist_ok=True)

        open_count = 0
        closed_count = 0

        # Find and organize images
        for img_path in tqdm(list(cew_path.rglob('*.jpg')) + list(cew_path.rglob('*.png')),
                            desc="Processing eye images"):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # Determine if open or closed based on filename or directory
                if 'closed' in str(img_path).lower() or '0' in img_path.stem[-1]:
                    dest = eye_closed_dir / img_path.name
                    closed_count += 1
                else:
                    dest = eye_open_dir / img_path.name
                    open_count += 1

                if not dest.exists():
                    shutil.copy(str(img_path), str(dest))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        # Enrich with labeled samples from extracted datasets (if available)
        for ds_name in ('driver_distraction', 'drowsy_driver'):
            ds_root = self.raw_dir / ds_name
            added = self._import_labeled_dataset_to_eye_yawn(
                ds_root,
                ds_name,
                eye_open_dir,
                eye_closed_dir,
                yawn_dir,
                non_yawn_dir,
            )
            open_count += added['eye_open']
            closed_count += added['eye_closed']

        print(f"\n✓ Eye Dataset Organized:")
        print(f"  Open eyes: {open_count} images")
        print(f"  Closed eyes: {closed_count} images")

        return open_count > 0 and closed_count > 0

    def organize_yawn_dataset(self):
        """Organize yawn dataset using only new sources (Kaggle + labeled extracted datasets)."""
        print("\n" + "=" * 70)
        print("ORGANIZING YAWN DATASET")
        print("=" * 70)

        yawn_dir = YAWN_DATASET_PATH / 'yawn'
        non_yawn_dir = YAWN_DATASET_PATH / 'non_yawn'

        # Required by user: delete old yawn dataset and rebuild from new sources only
        if yawn_dir.exists():
            shutil.rmtree(yawn_dir)
        if non_yawn_dir.exists():
            shutil.rmtree(non_yawn_dir)

        yawn_dir.mkdir(parents=True, exist_ok=True)
        non_yawn_dir.mkdir(parents=True, exist_ok=True)

        yawn_count = 0
        non_yawn_count = 0

        # Import dedicated Kaggle yawn dataset if downloaded
        yawn_kaggle_path = self.raw_dir / 'yawn_kaggle'
        if yawn_kaggle_path.exists():
            kaggle_images = self._iter_files(yawn_kaggle_path, ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'))
            for img_path in tqdm(kaggle_images, desc="Processing Kaggle yawn images"):
                lower_name = str(img_path).lower()
                relative_path = str(img_path.relative_to(yawn_kaggle_path)).lower()
                try:
                    if 'yawn' in lower_name and 'non' not in lower_name and 'no_yawn' not in lower_name:
                        dst = yawn_dir / f"kaggle_{img_path.name}"
                        self._copy_if_missing(img_path, dst)
                        yawn_count += 1
                    elif any(tag in lower_name for tag in ['non_yawn', 'noyawn', 'normal', 'not_yawn']) or 'non' in relative_path:
                        dst = non_yawn_dir / f"kaggle_{img_path.name}"
                        self._copy_if_missing(img_path, dst)
                        non_yawn_count += 1
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        else:
            print(f"⚠️  Kaggle yawn dataset not found at {yawn_kaggle_path}")
            print("   Download it using: kagglehub.dataset_download('davidvazquezcic/yawn-dataset')")

        # Enrich with CSV-labeled yawn data from extracted datasets
        for ds_name in ('driver_distraction', 'drowsy_driver'):
            ds_root = self.raw_dir / ds_name
            added = self._import_labeled_dataset_to_eye_yawn(
                ds_root,
                ds_name,
                EYE_DATASET_PATH / 'open',
                EYE_DATASET_PATH / 'closed',
                yawn_dir,
                non_yawn_dir,
            )
            yawn_count += added['yawn']
            non_yawn_count += added['non_yawn']

        print(f"\n✓ Yawn Dataset Organized:")
        total_yawn_images = self._count_images(yawn_dir)
        total_non_yawn_images = self._count_images(non_yawn_dir)
        print(f"  New yawn images added: {yawn_count}")
        print(f"  New non-yawn images added: {non_yawn_count}")
        print(f"  Total yawn images available: {total_yawn_images}")
        print(f"  Total non-yawn images available: {total_non_yawn_images}")

        return total_yawn_images > 0 and total_non_yawn_images > 0

    def organize_evaluation_dataset(self):
        """Copy evaluation videos to evaluation_videos directory."""
        print("\n" + "=" * 70)
        print("ORGANIZING EVALUATION DATASET")
        print("=" * 70)

        EVALUATION_VIDEO_PATH.mkdir(parents=True, exist_ok=True)

        if not self.raw_dir.exists():
            print(f"⚠️  Raw dataset directory not found at {self.raw_dir}")
            print("Skipping evaluation dataset organization")
            return False

        video_count = 0

        # Prefer directly available videos from all raw datasets (YawDD has many)
        video_files = self._iter_files(
            self.raw_dir,
            ('*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4', '*.AVI', '*.MOV', '*.MKV')
        )

        for video_path in tqdm(video_files, desc="Copying evaluation videos"):
            try:
                dest = EVALUATION_VIDEO_PATH / video_path.name
                if not dest.exists():
                    shutil.copy(str(video_path), str(dest))
                    video_count += 1
            except Exception as e:
                print(f"Error copying {video_path}: {e}")

        print(f"\n✓ Evaluation Dataset Organized:")
        total_videos = len(list(EVALUATION_VIDEO_PATH.glob('*.mp4'))) + len(list(EVALUATION_VIDEO_PATH.glob('*.avi')))
        print(f"  New videos copied: {video_count} files")
        print(f"  Total videos available: {total_videos} files")

        return total_videos > 0


class ModelTrainer:
    """Handles model training and evaluation."""

    def __init__(self):
        self.results = {}

    def plot_confusion_matrix(self, y_true, y_pred, classes, title, save_path):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved: {save_path}")

    def plot_training_history(self, history, model_name, save_path):
        """Plot and save training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(history.history['accuracy'], label='Train Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
        ax1.set_title(f'{model_name} - Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(history.history['loss'], label='Train Loss')
        ax2.plot(history.history['val_loss'], label='Val Loss')
        ax2.set_title(f'{model_name} - Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Training history saved: {save_path}")

    def _build_classifier_model(self, num_classes, learning_rate=2e-4):
        """Create transfer-learning classifier with MobileNetV2 backbone."""
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(CNN_INPUT_SIZE, CNN_INPUT_SIZE, 3),
            include_top=False,
            weights='imagenet',
        )
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(CNN_INPUT_SIZE, CNN_INPUT_SIZE, 3)),
            tf.keras.layers.Rescaling(2.0, offset=-1.0),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(num_classes, activation='softmax'),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        return model

    def _load_csv_labeled_dataset(self, dataset_root: Path, label_columns, max_per_class=3000,
                                  strict_single_label=False, exclude_drowsy_flags=False):
        """Load image tensors + labels from *_classes.csv based datasets."""
        images = []
        labels = []

        all_rows = []
        for csv_path in dataset_root.rglob('_classes.csv'):
            split_dir = csv_path.parent
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if 'filename' not in df.columns:
                continue
            all_rows.append((split_dir, df))

        if not all_rows:
            return np.array([]), np.array([])

        per_class_count = {i: 0 for i in range(len(label_columns))}

        for split_dir, df in all_rows:
            # Optional distraction-specific drowsy columns for de-noising
            drowsy_cols = [
                ' d0 - Eyes Closed',
                ' d1 - Yawning',
                ' d2 - Nodding Off',
            ]

            for _, row in df.iterrows():
                filename = str(row.get('filename', '')).strip()
                if not filename:
                    continue

                src = DatasetManager._resolve_image_path(dataset_root, split_dir, filename)
                if src is None:
                    continue

                active_labels = []
                for idx, col in enumerate(label_columns):
                    if col in df.columns:
                        try:
                            if int(float(row.get(col, 0))) == 1:
                                active_labels.append(idx)
                        except Exception:
                            pass

                if not active_labels:
                    continue

                if strict_single_label and len(active_labels) != 1:
                    continue

                if exclude_drowsy_flags:
                    has_drowsy_flags = False
                    for col in drowsy_cols:
                        if col in df.columns:
                            try:
                                if int(float(row.get(col, 0))) == 1:
                                    has_drowsy_flags = True
                                    break
                            except Exception:
                                continue
                    if has_drowsy_flags:
                        continue

                class_idx = active_labels[0]
                if per_class_count[class_idx] >= max_per_class:
                    continue

                img = cv2.imread(str(src))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))

                images.append(img)
                labels.append(class_idx)
                per_class_count[class_idx] += 1

        return np.array(images), np.array(labels)

    def _load_combined_drowsiness_dataset(self, max_per_class=5000):
        """Build a cleaner drowsiness dataset using drowsy_driver + driver_distraction d-labels."""
        return self._load_configurable_drowsiness_dataset(
            max_per_class=max_per_class,
            use_primary=True,
            use_auxiliary=True,
        )

    def _load_configurable_drowsiness_dataset(self, max_per_class=5000, use_primary=True, use_auxiliary=True):
        """Build drowsiness dataset from configurable sources."""
        images = []
        labels = []
        per_class_count = {0: 0, 1: 0}

        if use_primary:
            # Source 1: drowsy_driver (primary)
            drowsy_root = Path(__file__).parent / 'dataset' / 'raw' / 'drowsy_driver'
            for csv_path in drowsy_root.rglob('_classes.csv'):
                split_dir = csv_path.parent
                try:
                    df = pd.read_csv(csv_path)
                except Exception:
                    continue
                if 'filename' not in df.columns or 'Drowsyness' not in df.columns:
                    continue

                for _, row in df.iterrows():
                    filename = str(row.get('filename', '')).strip()
                    if not filename:
                        continue

                    try:
                        drowsy_label = int(float(row.get('Drowsyness', 0)))
                    except Exception:
                        drowsy_label = 0

                    label = 1 if drowsy_label == 1 else 0
                    if per_class_count[label] >= max_per_class:
                        continue

                    src = DatasetManager._resolve_image_path(drowsy_root, split_dir, filename)
                    if src is None:
                        continue

                    img = cv2.imread(str(src))
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))

                    images.append(img)
                    labels.append(label)
                    per_class_count[label] += 1

        if use_auxiliary:
            # Source 2: driver_distraction d-labels (auxiliary)
            aux_root = Path(__file__).parent / 'dataset' / 'raw' / 'driver_distraction'
            d0_col = ' d0 - Eyes Closed'
            d1_col = ' d1 - Yawning'
            d2_col = ' d2 - Nodding Off'
            d3_col = ' d3 - Eyes Open'

            for csv_path in aux_root.rglob('_classes.csv'):
                split_dir = csv_path.parent
                try:
                    df = pd.read_csv(csv_path)
                except Exception:
                    continue
                if 'filename' not in df.columns:
                    continue

                available = set(df.columns)
                if not {d0_col, d1_col, d2_col, d3_col}.issubset(available):
                    continue

                for _, row in df.iterrows():
                    filename = str(row.get('filename', '')).strip()
                    if not filename:
                        continue

                    try:
                        d0 = int(float(row.get(d0_col, 0)))
                        d1 = int(float(row.get(d1_col, 0)))
                        d2 = int(float(row.get(d2_col, 0)))
                        d3 = int(float(row.get(d3_col, 0)))
                    except Exception:
                        continue

                    # High-confidence mapping only
                    label = None
                    if (d0 == 1 or d1 == 1 or d2 == 1) and d3 == 0:
                        label = 1
                    elif d3 == 1 and d0 == 0 and d1 == 0 and d2 == 0:
                        label = 0

                    if label is None or per_class_count[label] >= max_per_class:
                        continue

                    src = DatasetManager._resolve_image_path(aux_root, split_dir, filename)
                    if src is None:
                        continue

                    img = cv2.imread(str(src))
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))

                    images.append(img)
                    labels.append(label)
                    per_class_count[label] += 1

        return np.array(images), np.array(labels)

    def _load_binary_distraction_dataset(self, dataset_root: Path, max_per_class=2500):
        """Build binary distraction dataset: safe vs distracted using c-labels only."""
        safe_col = ' c0 - Safe Driving'
        distraction_cols = [
            ' c1 - Texting',
            ' c2 - Talking on the phone',
            ' c3 - Operating the Radio',
            ' c4 - Drinking',
            ' c5 - Reaching Behind',
            ' c6 - Hair and Makeup',
            ' c7 - Talking to Passenger',
        ]
        drowsy_cols = [' d0 - Eyes Closed', ' d1 - Yawning', ' d2 - Nodding Off']

        images = []
        labels = []
        per_class_count = {0: 0, 1: 0}  # 0=safe, 1=distracted

        for csv_path in dataset_root.rglob('_classes.csv'):
            split_dir = csv_path.parent
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if 'filename' not in df.columns:
                continue

            for _, row in df.iterrows():
                filename = str(row.get('filename', '')).strip()
                if not filename:
                    continue

                has_drowsy_flags = False
                for col in drowsy_cols:
                    if col in df.columns:
                        try:
                            if int(float(row.get(col, 0))) == 1:
                                has_drowsy_flags = True
                                break
                        except Exception:
                            continue
                if has_drowsy_flags:
                    continue

                try:
                    safe_val = int(float(row.get(safe_col, 0)))
                    distraction_sum = sum(int(float(row.get(col, 0))) for col in distraction_cols)
                except Exception:
                    continue

                label = None
                if safe_val == 1 and distraction_sum == 0:
                    label = 0
                elif safe_val == 0 and distraction_sum >= 1:
                    label = 1

                if label is None or per_class_count[label] >= max_per_class:
                    continue

                src = DatasetManager._resolve_image_path(dataset_root, split_dir, filename)
                if src is None:
                    continue

                img = cv2.imread(str(src))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))

                images.append(img)
                labels.append(label)
                per_class_count[label] += 1

        return np.array(images), np.array(labels)

    def train_distraction_model(self):
        """Train a separate multi-class driver distraction model (c0..c7)."""
        print("\n" + "=" * 70)
        print("TRAINING DRIVER DISTRACTION MODEL")
        print("=" * 70)

        dataset_root = Path(__file__).parent / 'dataset' / 'raw' / 'driver_distraction'
        class_names = ['safe', 'distracted']
        images, labels = self._load_binary_distraction_dataset(dataset_root, max_per_class=3000)
        if len(images) == 0:
            print("✗ No distraction training data found")
            return False

        print(f"✓ Loaded {len(images)} distraction images")
        distribution = [(name, int(np.sum(labels == idx))) for idx, name in enumerate(class_names)]
        print(f"  Distribution: {distribution}")

        data_loader = DataLoader(img_size=CNN_INPUT_SIZE, batch_size=CNN_BATCH_SIZE)
        train_ds, val_ds, test_ds = data_loader.prepare_dataset(images, labels)
        train_ds = data_loader.apply_augmentation(train_ds)

        model = self._build_classifier_model(num_classes=len(class_names), learning_rate=2e-4)
        class_weights = {}
        for idx in range(len(class_names)):
            count = max(int(np.sum(labels == idx)), 1)
            class_weights[idx] = float(len(labels) / (len(class_names) * count))

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=min(CNN_EPOCHS, 20),
            class_weight=class_weights,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True,
                    mode='max',
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=2,
                    min_lr=1e-6,
                    verbose=0,
                ),
            ],
        )
        loss, accuracy = model.evaluate(test_ds)

        y_pred = []
        y_true = []
        for batch_images, batch_labels in test_ds:
            pred = model.predict(batch_images, verbose=0)
            y_pred.extend(np.argmax(pred, axis=1))
            y_true.extend(batch_labels.numpy())

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        model.save(str(DISTRACTION_MODEL_PATH))
        self.plot_training_history(history, "Driver Distraction", str(RESULTS_DIR / "distraction_model_history.png"))
        self.plot_confusion_matrix(y_true, y_pred, class_names, "Driver Distraction - Confusion Matrix", str(RESULTS_DIR / "distraction_model_confusion_matrix.png"))

        self.results['distraction'] = {
            'accuracy': float(accuracy),
            'loss': float(loss),
            'classes': class_names,
            'samples': int(len(images)),
        }
        print("\n✓ Distraction Model Training Completed!")
        return True

    def train_drowsiness_model(self):
        """Train a separate binary drowsiness model from drowsy_driver dataset."""
        print("\n" + "=" * 70)
        print("TRAINING DROWSINESS MODEL")
        print("=" * 70)

        images, labels = self._load_configurable_drowsiness_dataset(
            max_per_class=5000,
            use_primary=False,
            use_auxiliary=True,
        )

        if len(images) == 0:
            print("✗ No drowsiness training data found")
            return False

        print(f"✓ Loaded {len(images)} drowsiness images")
        print(f"  Distribution: [('awake', {int(np.sum(labels == 0))}), ('drowsy', {int(np.sum(labels == 1))})]")

        data_loader = DataLoader(img_size=CNN_INPUT_SIZE, batch_size=CNN_BATCH_SIZE)
        train_ds, val_ds, test_ds = data_loader.prepare_dataset(images, labels)
        train_ds = data_loader.apply_augmentation(train_ds)

        model = self._build_classifier_model(num_classes=2, learning_rate=2e-4)
        class_weights = {
            0: float(len(labels) / (2 * max(int(np.sum(labels == 0)), 1))),
            1: float(len(labels) / (2 * max(int(np.sum(labels == 1)), 1))),
        }
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=min(CNN_EPOCHS, 20),
            class_weight=class_weights,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True,
                    mode='max',
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=2,
                    min_lr=1e-6,
                    verbose=0,
                ),
            ],
        )
        loss, accuracy = model.evaluate(test_ds)

        y_pred = []
        y_true = []
        for batch_images, batch_labels in test_ds:
            pred = model.predict(batch_images, verbose=0)
            y_pred.extend(np.argmax(pred, axis=1))
            y_true.extend(batch_labels.numpy())

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        model.save(str(DROWSINESS_MODEL_PATH))
        self.plot_training_history(history, "Drowsiness", str(RESULTS_DIR / "drowsiness_model_history.png"))
        self.plot_confusion_matrix(y_true, y_pred, ['awake', 'drowsy'], "Drowsiness - Confusion Matrix", str(RESULTS_DIR / "drowsiness_model_confusion_matrix.png"))

        self.results['drowsiness'] = {
            'accuracy': float(accuracy),
            'loss': float(loss),
            'classes': ['awake', 'drowsy'],
            'samples': int(len(images)),
        }
        print("\n✓ Drowsiness Model Training Completed!")
        return True

    def train_eye_model(self):
        """Train eye state detection model."""
        print("\n" + "=" * 70)
        print("TRAINING EYE STATE DETECTION MODEL")
        print("=" * 70)

        if not EYE_DATASET_PATH.exists():
            print(f"✗ Eye dataset path does not exist: {EYE_DATASET_PATH}")
            return False

        print(f"\nLoading eye dataset from: {EYE_DATASET_PATH}")

        try:
            data_loader = DataLoader(img_size=CNN_INPUT_SIZE, batch_size=CNN_BATCH_SIZE)

            # Load dataset
            eye_images, eye_labels, eye_classes = data_loader.load_dataset_from_folders({
                'open': str(EYE_DATASET_PATH / 'open'),
                'closed': str(EYE_DATASET_PATH / 'closed'),
            })

            print(f"✓ Loaded {len(eye_images)} eye images")
            print(f"  Classes: {eye_classes}")
            print(f"  Distribution: {[(c, sum(eye_labels == i)) for i, c in enumerate(eye_classes)]}")

            # Prepare dataset
            print("\nPreparing dataset (train: 70%, val: 15%, test: 15%)...")
            train_ds, val_ds, test_ds = data_loader.prepare_dataset(eye_images, eye_labels)

            # Apply augmentation
            print("Applying data augmentation...")
            train_ds = data_loader.apply_augmentation(train_ds)

            # Create model
            print("\nCreating Eye State Model (MobileNetV2)...")
            model = EyeStateModel(CNN_INPUT_SIZE)
            model.compile(learning_rate=CNN_LEARNING_RATE)

            # Train
            print(f"Training ({CNN_EPOCHS} epochs, batch_size={CNN_BATCH_SIZE})...")
            history = model.train(train_ds, val_ds, epochs=CNN_EPOCHS)

            # Evaluate
            print("\nEvaluating on test set...")
            loss, accuracy = model.evaluate(test_ds)
            print(f"Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f}")

            # Get predictions for confusion matrix
            y_pred = []
            y_true = []
            for images, labels in test_ds:
                pred = model.model.predict(images, verbose=0)
                y_pred.extend(np.argmax(pred, axis=1))
                y_true.extend(labels.numpy())

            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=eye_classes))

            # Save model
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            print(f"\nSaving Eye Model to {EYE_MODEL_PATH}...")
            model.save(str(EYE_MODEL_PATH))

            # Plot history
            history_path = RESULTS_DIR / "eye_model_history.png"
            self.plot_training_history(history, "Eye State Detection", str(history_path))

            # Plot confusion matrix
            cm_path = RESULTS_DIR / "eye_model_confusion_matrix.png"
            self.plot_confusion_matrix(y_true, y_pred, eye_classes, "Eye State - Confusion Matrix", str(cm_path))

            self.results['eye'] = {
                'accuracy': float(accuracy),
                'loss': float(loss),
                'classes': eye_classes,
                'samples': len(eye_images),
            }

            print("\n✓ Eye Model Training Completed!")
            return True

        except Exception as e:
            print(f"\n✗ Error during eye model training: {e}")
            import traceback
            traceback.print_exc()
            return False

    def train_yawn_model(self):
        """Train yawn detection model."""
        print("\n" + "=" * 70)
        print("TRAINING YAWN DETECTION MODEL")
        print("=" * 70)

        if not YAWN_DATASET_PATH.exists():
            print(f"✗ Yawn dataset path does not exist: {YAWN_DATASET_PATH}")
            return False

        print(f"\nLoading yawn dataset from: {YAWN_DATASET_PATH}")

        try:
            data_loader = DataLoader(img_size=CNN_INPUT_SIZE, batch_size=CNN_BATCH_SIZE)

            # Load dataset
            yawn_images, yawn_labels, yawn_classes = data_loader.load_dataset_from_folders({
                'yawn': str(YAWN_DATASET_PATH / 'yawn'),
                'non_yawn': str(YAWN_DATASET_PATH / 'non_yawn'),
            })

            print(f"✓ Loaded {len(yawn_images)} yawn images")
            print(f"  Classes: {yawn_classes}")
            print(f"  Distribution: {[(c, sum(yawn_labels == i)) for i, c in enumerate(yawn_classes)]}")

            # Prepare dataset
            print("\nPreparing dataset (train: 70%, val: 15%, test: 15%)...")
            train_ds, val_ds, test_ds = data_loader.prepare_dataset(yawn_images, yawn_labels)

            # Apply augmentation
            print("Applying data augmentation...")
            train_ds = data_loader.apply_augmentation(train_ds)

            # Create model
            print("\nCreating Yawn Detection Model (MobileNetV2)...")
            model = YawDetectionModel(CNN_INPUT_SIZE)
            model.compile(learning_rate=CNN_LEARNING_RATE)

            # Train
            print(f"Training ({CNN_EPOCHS} epochs, batch_size={CNN_BATCH_SIZE})...")
            history = model.train(train_ds, val_ds, epochs=CNN_EPOCHS)

            # Evaluate
            print("\nEvaluating on test set...")
            loss, accuracy = model.evaluate(test_ds)
            print(f"Test Loss: {loss:.4f} | Test Accuracy: {accuracy:.4f}")

            # Get predictions for confusion matrix
            y_pred = []
            y_true = []
            for images, labels in test_ds:
                pred = model.model.predict(images, verbose=0)
                y_pred.extend(np.argmax(pred, axis=1))
                y_true.extend(labels.numpy())

            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=yawn_classes))

            # Save model
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            print(f"\nSaving Yawn Model to {YAWN_MODEL_PATH}...")
            model.save(str(YAWN_MODEL_PATH))

            # Plot history
            history_path = RESULTS_DIR / "yawn_model_history.png"
            self.plot_training_history(history, "Yawn Detection", str(history_path))

            # Plot confusion matrix
            cm_path = RESULTS_DIR / "yawn_model_confusion_matrix.png"
            self.plot_confusion_matrix(y_true, y_pred, yawn_classes, "Yawn Detection - Confusion Matrix", str(cm_path))

            self.results['yawn'] = {
                'accuracy': float(accuracy),
                'loss': float(loss),
                'classes': yawn_classes,
                'samples': len(yawn_images),
            }

            print("\n✓ Yawn Model Training Completed!")
            return True

        except Exception as e:
            print(f"\n✗ Error during yawn model training: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_results(self):
        """Save training results to JSON."""
        results_file = RESULTS_DIR / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Complete setup and model training pipeline")
    parser.add_argument('--skip-download', action='store_true', help='Skip Kaggle dataset download')
    parser.add_argument('--download-only', action='store_true', help='Only download/organize datasets, skip training')
    parser.add_argument('--train-only', action='store_true', help='Skip download and run training only')
    parser.add_argument('--skip-eye', action='store_true', help='Skip eye model training')
    parser.add_argument('--skip-yawn', action='store_true', help='Skip yawn model training')
    parser.add_argument('--skip-distraction', action='store_true', help='Skip distraction model training')
    parser.add_argument('--skip-drowsiness', action='store_true', help='Skip drowsiness model training')
    args = parser.parse_args()

    if args.download_only and args.train_only:
        print("\n✗ Invalid options: --download-only and --train-only cannot be used together")
        return 2

    skip_download = args.skip_download or args.train_only

    print("\n" + "=" * 70)
    print("DRIVER FATIGUE MONITORING SYSTEM - SETUP & TRAINING")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Step 1: Download datasets
    dataset_manager = DatasetManager()

    if not skip_download:
        dataset_manager.download_datasets()
    else:
        print("\nSkipping dataset download (--skip-download flag)")

    # Step 2: Organize datasets
    print("\nOrganizing datasets into required structure...")
    eye_ready = dataset_manager.organize_eye_dataset()
    yawn_ready = dataset_manager.organize_yawn_dataset()
    eval_ready = dataset_manager.organize_evaluation_dataset()

    # Step 3: Train models
    trainer = ModelTrainer()

    if args.download_only:
        train_eye_success = False
        train_yawn_success = False
        train_distraction_success = False
        train_drowsiness_success = False
        print("\nSkipping model training (--download-only flag)")
    elif eye_ready and not args.skip_eye:
        train_eye_success = trainer.train_eye_model()
    else:
        reason = "flag enabled" if args.skip_eye else "dataset not ready"
        print(f"\n⚠️  Skipping eye model training - {reason}")
        train_eye_success = False

    if not args.download_only and yawn_ready and not args.skip_yawn:
        train_yawn_success = trainer.train_yawn_model()
    else:
        reason = "flag enabled" if args.skip_yawn else "dataset not ready"
        print(f"\n⚠️  Skipping yawn model training - {reason}")
        train_yawn_success = False

    if not args.download_only and not args.skip_distraction:
        train_distraction_success = trainer.train_distraction_model()
    else:
        train_distraction_success = False
        if args.skip_distraction:
            print("\n⚠️  Skipping distraction model training - flag enabled")

    if not args.download_only and not args.skip_drowsiness:
        train_drowsiness_success = trainer.train_drowsiness_model()
    else:
        train_drowsiness_success = False
        if args.skip_drowsiness:
            print("\n⚠️  Skipping drowsiness model training - flag enabled")

    # Step 4: Save results
    trainer.save_results()

    # Step 5: Final summary
    print("\n" + "=" * 70)
    print("SETUP & TRAINING SUMMARY")
    print("=" * 70)
    print(f"\nDataset Status:")
    print(f"  Eye Dataset: {'✓ Ready' if eye_ready else '✗ Not Ready'}")
    print(f"  Yawn Dataset: {'✓ Ready' if yawn_ready else '✗ Not Ready'}")
    print(f"  Evaluation Dataset: {'✓ Ready' if eval_ready else '✗ Not Ready'}")

    print(f"\nModel Training Status:")
    print(f"  Eye Model: {'✓ SUCCESS' if train_eye_success else '✗ FAILED'}")
    print(f"  Yawn Model: {'✓ SUCCESS' if train_yawn_success else '✗ FAILED'}")
    print(f"  Distraction Model: {'✓ SUCCESS' if train_distraction_success else '✗ FAILED'}")
    print(f"  Drowsiness Model: {'✓ SUCCESS' if train_drowsiness_success else '✗ FAILED'}")

    if args.download_only:
        print(f"\n✓ Dataset setup completed successfully.")
        print(f"Run training later with:")
        print(f"  python setup_and_train.py --train-only")
        return 0

    if train_eye_success and train_yawn_success and train_distraction_success and train_drowsiness_success:
        print(f"\n✓ All models trained successfully!")
        print(f"\nYou can now run the full monitoring system:")
        print(f"  python src/main.py --mode webcam")
        print(f"\nOr process evaluation videos:")
        print(f"  python src/main.py --mode video --video {EVALUATION_VIDEO_PATH}/your_video.mp4")
        return 0
    else:
        print(f"\n⚠️  Some models failed to train.")
        print(f"Please check the errors above and ensure datasets are properly downloaded.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
