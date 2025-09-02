#!/usr/bin/env python3
"""
Train Plant Disease Detection Model

This script trains a CNN model for plant disease classification using
proper validation, callbacks, and saves a fully trained model.
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from src.models.architectures import ModelBuilder
from src.data.dataloader import PlantDiseaseDataLoader


def create_callbacks(output_dir: Path):
    """Create training callbacks."""
    callbacks = []
    
    # Model checkpoint
    checkpoint_path = output_dir / "checkpoints" / "best_model.h5"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks.append(ModelCheckpoint(
        str(checkpoint_path),
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    ))
    
    # Early stopping
    callbacks.append(EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ))
    
    # Reduce learning rate
    callbacks.append(ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ))
    
    # TensorBoard
    log_dir = output_dir / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks.append(TensorBoard(log_dir=str(log_dir), histogram_freq=1))
    
    return callbacks


def train_model(data_dir: str, output_dir: str, epochs: int = 50, batch_size: int = 32):
    """Train the plant disease detection model."""
    print("Starting Plant Disease Detection Model Training")
    
    # Set paths
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize data loader
    print("\nLoading dataset...")
    data_loader = PlantDiseaseDataLoader(
        data_dir=str(data_path),
        image_size=(224, 224),
        batch_size=batch_size
    )
    
    # Create data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Load data
    train_generator = train_datagen.flow_from_directory(
        data_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        data_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    # Get class information
    num_classes = len(train_generator.class_indices)
    class_names = {v: k for k, v in train_generator.class_indices.items()}
    
    print(f"\nDataset Statistics:")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Training samples: {train_generator.samples}")
    print(f"  - Validation samples: {val_generator.samples}")
    
    # Save class names
    class_names_path = output_path / "class_names.json"
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"\nSaved class names to {class_names_path}")
    
    # Build model
    print("\nBuilding model architecture...")
    builder = ModelBuilder(num_classes=num_classes, input_shape=(224, 224, 3))
    model = builder.build_model(
        architecture="custom-cnn",
        pretrained=False,
        dropout_rate=0.3
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks(output_path)
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = output_path / "final_model.h5"
    model.save(final_model_path)
    print(f"\nSaved final model to {final_model_path}")
    
    # Save training history
    history_path = output_path / "training_history.json"
    history_dict = {
        key: [float(val) for val in values] 
        for key, values in history.history.items()
    }
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"Saved training history to {history_path}")
    
    # Evaluate on validation set
    print("\nFinal Evaluation:")
    val_loss, val_acc, val_top3_acc = model.evaluate(val_generator, verbose=0)
    print(f"  - Validation Loss: {val_loss:.4f}")
    print(f"  - Validation Accuracy: {val_acc:.4f}")
    print(f"  - Validation Top-3 Accuracy: {val_top3_acc:.4f}")
    
    # Save model for deployment
    deploy_dir = Path("weights/pretrained")
    deploy_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy best model
    best_checkpoint = output_path / "checkpoints" / "best_model.h5"
    if best_checkpoint.exists():
        import shutil
        shutil.copy(best_checkpoint, deploy_dir / "best_model.h5")
        shutil.copy(class_names_path, deploy_dir / "class_names.json")
        print(f"\nDeployed model to {deploy_dir}")
    
    print("\nTraining completed successfully!")
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train Plant Disease Detection Model")
    parser.add_argument(
        "--data-dir", 
        type=str, 
        required=True,
        help="Path to dataset directory (should contain class subdirectories)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="training_output",
        help="Directory to save training outputs"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32,
        help="Batch size for training"
    )
    
    args = parser.parse_args()
    
    # Train model
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()