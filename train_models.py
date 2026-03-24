#!/usr/bin/env python3
"""
Training script for CNN models (Eye State and Yawn Detection).

Usage:
    python train_models.py --eye_only          # Train only eye model
    python train_models.py --yawn_only         # Train only yawn model
    python train_models.py                     # Train both models
"""

import os
import sys
import argparse
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training import DataLoader, EyeStateModel, YawDetectionModel
from src.config import (
    CNN_INPUT_SIZE, CNN_BATCH_SIZE, CNN_EPOCHS, CNN_LEARNING_RATE,
    EYE_DATASET_PATH, YAWN_DATASET_PATH, EYE_MODEL_PATH, YAWN_MODEL_PATH,
    RESULTS_DIR
)


def plot_training_history(history, model_name, save_path):
    """Plot and save training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title(f'{model_name} - Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    # Loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title(f'{model_name} - Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Training history saved to {save_path}")


def train_eye_model(data_loader, epochs, learning_rate, use_augmentation=True):
    """Train eye state detection model."""
    print("\n" + "=" * 60)
    print("TRAINING EYE STATE DETECTION MODEL")
    print("=" * 60)

    # Check if dataset exists
    if not EYE_DATASET_PATH.exists():
        print(f"ERROR: Eye dataset path does not exist: {EYE_DATASET_PATH}")
        print("Expected structure:")
        print("  dataset/eye/open/      # Images of open eyes")
        print("  dataset/eye/closed/    # Images of closed eyes")
        return False

    print(f"\nLoading eye dataset from: {EYE_DATASET_PATH}")

    try:
        # Load dataset
        eye_images, eye_labels, eye_classes = data_loader.load_dataset_from_folders({
            'open': str(EYE_DATASET_PATH / 'open'),
            'closed': str(EYE_DATASET_PATH / 'closed'),
        })

        print(f"Loaded {len(eye_images)} eye images")
        print(f"Classes: {eye_classes}")
        print(f"Class distribution: {[(class_name, sum(eye_labels == i)) for i, class_name in enumerate(eye_classes)]}")

        # Prepare dataset
        print("\nPreparing dataset (train/val/test split)...")
        train_ds, val_ds, test_ds = data_loader.prepare_dataset(eye_images, eye_labels)

        # Apply data augmentation to training data
        if use_augmentation:
            print("Applying data augmentation...")
            train_ds = data_loader.apply_augmentation(train_ds)

        # Create model
        print("\nCreating eye model (MobileNetV2)...")
        eye_model = EyeStateModel(CNN_INPUT_SIZE)
        eye_model.compile(learning_rate=learning_rate)

        print("Model architecture:")
        eye_model.model.summary()

        # Train
        print(f"\nTraining eye model ({epochs} epochs)...")
        history = eye_model.train(train_ds, val_ds, epochs=epochs)

        # Evaluate
        print("\nEvaluating eye model on test set...")
        loss, accuracy = eye_model.evaluate(test_ds)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

        # Save model
        print(f"\nSaving eye model to {EYE_MODEL_PATH}...")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        eye_model.save(str(EYE_MODEL_PATH))

        # Plot history
        plot_path = RESULTS_DIR / "eye_model_history.png"
        plot_training_history(history, "Eye State Detection", str(plot_path))

        print("\n✓ Eye model training completed successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Error during eye model training: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_yawn_model(data_loader, epochs, learning_rate, use_augmentation=True):
    """Train yawn detection model."""
    print("\n" + "=" * 60)
    print("TRAINING YAWN DETECTION MODEL")
    print("=" * 60)

    # Check if dataset exists
    if not YAWN_DATASET_PATH.exists():
        print(f"ERROR: Yawn dataset path does not exist: {YAWN_DATASET_PATH}")
        print("Expected structure:")
        print("  dataset/yawn/yawn/     # Images of yawning")
        print("  dataset/yawn/non_yawn/ # Images of not yawning")
        return False

    print(f"\nLoading yawn dataset from: {YAWN_DATASET_PATH}")

    try:
        # Load dataset
        yawn_images, yawn_labels, yawn_classes = data_loader.load_dataset_from_folders({
            'yawn': str(YAWN_DATASET_PATH / 'yawn'),
            'non_yawn': str(YAWN_DATASET_PATH / 'non_yawn'),
        })

        print(f"Loaded {len(yawn_images)} yawn images")
        print(f"Classes: {yawn_classes}")
        print(f"Class distribution: {[(class_name, sum(yawn_labels == i)) for i, class_name in enumerate(yawn_classes)]}")

        # Prepare dataset
        print("\nPreparing dataset (train/val/test split)...")
        train_ds, val_ds, test_ds = data_loader.prepare_dataset(yawn_images, yawn_labels)

        # Apply data augmentation
        if use_augmentation:
            print("Applying data augmentation...")
            train_ds = data_loader.apply_augmentation(train_ds)

        # Create model
        print("\nCreating yawn model (MobileNetV2)...")
        yawn_model = YawDetectionModel(CNN_INPUT_SIZE)
        yawn_model.compile(learning_rate=learning_rate)

        print("Model architecture:")
        yawn_model.model.summary()

        # Train
        print(f"\nTraining yawn model ({epochs} epochs)...")
        history = yawn_model.train(train_ds, val_ds, epochs=epochs)

        # Evaluate
        print("\nEvaluating yawn model on test set...")
        loss, accuracy = yawn_model.evaluate(test_ds)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

        # Save model
        print(f"\nSaving yawn model to {YAWN_MODEL_PATH}...")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        yawn_model.save(str(YAWN_MODEL_PATH))

        # Plot history
        plot_path = RESULTS_DIR / "yawn_model_history.png"
        plot_training_history(history, "Yawn Detection", str(plot_path))

        print("\n✓ Yawn model training completed successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Error during yawn model training: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train CNN models for driver fatigue detection"
    )
    parser.add_argument("--eye_only", action="store_true", help="Train only eye model")
    parser.add_argument("--yawn_only", action="store_true", help="Train only yawn model")
    parser.add_argument("--epochs", type=int, default=CNN_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=CNN_BATCH_SIZE, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=CNN_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--max_samples_per_class", type=int, default=2000,
                        help="Max images per class (set 0 to use all)")
    parser.add_argument("--no_augmentation", action="store_true", help="Disable data augmentation")

    args = parser.parse_args()

    # Create data loader
    max_samples = None if args.max_samples_per_class == 0 else args.max_samples_per_class
    data_loader = DataLoader(img_size=CNN_INPUT_SIZE, batch_size=args.batch_size, max_samples_per_class=max_samples)

    print("\n" + "=" * 60)
    print("DRIVER FATIGUE MONITORING - MODEL TRAINING")
    print("=" * 60)
    print(f"Input Size: {CNN_INPUT_SIZE}x{CNN_INPUT_SIZE}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.learning_rate}")

    # Train models
    results = {}

    if args.yawn_only:
        results['yawn'] = train_yawn_model(
            data_loader,
            args.epochs,
            args.learning_rate,
            use_augmentation=not args.no_augmentation,
        )
    elif args.eye_only:
        results['eye'] = train_eye_model(
            data_loader,
            args.epochs,
            args.learning_rate,
            use_augmentation=not args.no_augmentation,
        )
    else:
        # Train both
        results['eye'] = train_eye_model(
            data_loader,
            args.epochs,
            args.learning_rate,
            use_augmentation=not args.no_augmentation,
        )
        results['yawn'] = train_yawn_model(
            data_loader,
            args.epochs,
            args.learning_rate,
            use_augmentation=not args.no_augmentation,
        )

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    for model_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{model_name.upper()} Model: {status}")

    if all(results.values()):
        print("\n✓ All models trained successfully!")
        print("\nYou can now run the monitoring system:")
        print("  python src/main.py --mode webcam")
        return 0
    else:
        print("\n✗ Some models failed to train. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
