import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split


class DataLoader:
    """Loads and preprocesses image data for training."""

    def __init__(self, img_size=224, batch_size=32, max_samples_per_class=2000):
        self.img_size = img_size
        self.batch_size = batch_size
        self.max_samples_per_class = max_samples_per_class

    def load_dataset_from_folders(self, class_dirs):
        """
        Load images from class folders.
        class_dirs: dict with format {'class_name': 'path/to/class/folder'}
        Returns: (images, labels, class_names)
        """
        images = []
        labels = []
        class_names = list(class_dirs.keys())

        for class_idx, (class_name, folder_path) in enumerate(class_dirs.items()):
            folder = Path(folder_path)
            if not folder.exists():
                print(f"Warning: {folder_path} does not exist")
                continue

            image_files = []
            for pattern in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
                image_files.extend(folder.glob(pattern))

            if self.max_samples_per_class and len(image_files) > self.max_samples_per_class:
                original_count = len(image_files)
                rng = np.random.default_rng(42)
                selected_indices = rng.choice(
                    len(image_files),
                    size=self.max_samples_per_class,
                    replace=False,
                )
                image_files = [image_files[i] for i in selected_indices]
                print(
                    f"Sampling {self.max_samples_per_class} images for class '{class_name}' "
                    f"from {original_count}"
                )

            for img_file in image_files:
                try:
                    img = cv2.imread(str(img_file))
                    if img is None:
                        continue

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    images.append(img)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")

        if not images:
            raise ValueError("No images found in the specified directories")

        return np.array(images), np.array(labels), class_names

    def prepare_dataset(self, images, labels, test_size=0.2, val_size=0.2):
        """
        Split dataset into train, val, test and apply preprocessing.
        Returns: (train_ds, val_ds, test_ds)
        """
        # Normalize images
        images = images.astype(np.float32) / 255.0

        # Split into train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=42, stratify=labels
        )

        # Split train+val into train and val
        val_split = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_split, random_state=42, stratify=y_temp
        )

        # Convert to TensorFlow datasets
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_ds = train_ds.shuffle(buffer_size=1000).batch(self.batch_size)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_ds = val_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_ds = test_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds, test_ds

    def apply_augmentation(self, dataset):
        """Apply data augmentation to training dataset."""
        augmentation_layers = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.2),
        ])

        def augment_batch(x, y):
            return augmentation_layers(x)

        return dataset.map(lambda x, y: (augment_batch(x, y), y), num_parallel_calls=tf.data.AUTOTUNE)


class EyeStateModel:
    """CNN model for eye state classification (Open/Closed)."""

    def __init__(self, img_size=224):
        self.img_size = img_size
        self.model = self._build_model()

    def _build_model(self):
        """Build transfer learning model with MobileNetV2."""
        base_model = MobileNetV2(
            input_shape=(self.img_size, self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )

        # Fine-tune upper layers for stronger domain adaptation
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        # Add custom top layers
        model = keras.Sequential([
            layers.Input(shape=(self.img_size, self.img_size, 3)),
            layers.Rescaling(2.0, offset=-1.0),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(2, activation='softmax')  # Binary classification
        ])

        return model

    def compile(self, learning_rate=0.001):
        """Compile model."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, train_ds, val_ds, epochs=50, callbacks=None):
        """Train the model."""
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks if callbacks else []
        )
        return history

    def evaluate(self, test_ds):
        """Evaluate model on test set."""
        return self.model.evaluate(test_ds)

    def predict(self, image):
        """
        Predict eye state for a single image.
        image: numpy array of shape (224, 224, 3) with values in [0, 255]
        Returns: (class_idx, confidence)
        CLASS MAPPING: 0=OPEN (eyes are open), 1=CLOSED (eyes are closed)
        """
        if image is None or image.size == 0:
            return 0, 0.5  # Default to open if extraction failed
        
        image = image.astype(np.float32)
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        # Ensure in correct range
        image = np.clip(image, 0.0, 1.0)
        image = np.expand_dims(image, axis=0)
        
        prediction = self.model.predict(image, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])
        
        # Debug: log if confidence is uncertain
        if confidence < 0.55:
            # If uncertain, use aspect ratio as fallback signal
            pass
        
        return class_idx, confidence

    def save(self, path):
        """Save model."""
        self.model.save(path)

    def load(self, path):
        """Load model."""
        self.model = keras.models.load_model(path)


class YawDetectionModel:
    """CNN model for yawn detection (Yawn/Non-Yawn)."""

    def __init__(self, img_size=224):
        self.img_size = img_size
        self.model = self._build_model()

    def _build_model(self):
        """Build transfer learning model with MobileNetV2."""
        base_model = MobileNetV2(
            input_shape=(self.img_size, self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )

        # Fine-tune upper layers for better domain adaptation on yawn dataset
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        # Add custom top layers
        model = keras.Sequential([
            layers.Input(shape=(self.img_size, self.img_size, 3)),
            layers.Rescaling(2.0, offset=-1.0),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(2, activation='softmax')  # Binary classification
        ])

        return model

    def compile(self, learning_rate=0.001):
        """Compile model."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, train_ds, val_ds, epochs=50, callbacks=None):
        """Train the model."""
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks if callbacks else []
        )
        return history

    def evaluate(self, test_ds):
        """Evaluate model on test set."""
        return self.model.evaluate(test_ds)

    def predict(self, image):
        """
        Predict yawn state for a single image.
        image: numpy array of shape (224, 224, 3) with values in [0, 255]
        Returns: (class_idx, confidence)
        CLASS MAPPING: 0=YAWNING (mouth open in yawn), 1=NOT_YAWNING (normal mouth)
        """
        if image is None or image.size == 0:
            return 1, 0.5  # Default to not yawning if extraction failed
        
        image = image.astype(np.float32)
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        # Ensure in correct range
        image = np.clip(image, 0.0, 1.0)
        image = np.expand_dims(image, axis=0)
        
        prediction = self.model.predict(image, verbose=0)
        class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][class_idx])
        
        return class_idx, confidence

    def save(self, path):
        """Save model."""
        self.model.save(path)

    def load(self, path):
        """Load model."""
        self.model = keras.models.load_model(path)
