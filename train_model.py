"""
Production-Ready Mask Detection Model
Optimized for real-world deployment with face detection integration
"""

import json
import math
from pathlib import Path
from typing import Tuple, Dict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ============================================================
# CONFIGURATION
# ============================================================

# Model Parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_PHASE1 = 30
EPOCHS_PHASE2 = 20
RANDOM_SEED = 42

# Learning Rates
INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-5
MIN_LR = 1e-7

# Regularization
DROPOUT_RATE = 0.3
L2_WEIGHT = 1e-5
LABEL_SMOOTHING = 0.1

# Data Paths
DATA_ROOT = Path('data')
TRAIN_DIR = DATA_ROOT / 'train'
VAL_DIR = DATA_ROOT / 'Validation'
TEST_DIR = DATA_ROOT / 'test'

# Output Paths
MODEL_DIR = Path('models')
MODEL_H5_PATH = MODEL_DIR / 'mask_detector_production.h5'
PLOT_PATH = MODEL_DIR / 'training_history_production.png'
METADATA_PATH = MODEL_DIR / 'model_metadata.json'

# Class Configuration
CANONICAL_CLASSES = ['with_mask', 'without_mask']
CLASS_ALIASES = {
    'with_mask': ['with_mask', 'WithMask', 'mask', 'Mask'],
    'without_mask': ['without_mask', 'WithoutMask', 'nomask', 'NoMask'],
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def setup_runtime() -> None:
    """Initialize runtime environment with optimizations"""
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    # GPU Configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f'✓ GPU available: {len(gpus)} device(s)')
        except RuntimeError as e:
            print(f'⚠ GPU configuration warning: {e}')
    else:
        print('⚠ No GPU detected, using CPU')
    
    # Set mixed precision for faster training
    try:
        if gpus:
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
            print('✓ Mixed precision (float16) enabled')
    except Exception as e:
        print(f'⚠ Mixed precision not enabled: {e}')


def ensure_dataset_layout(split_dir: Path) -> None:
    """Organize dataset into canonical class structure"""
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # Create canonical class directories
    for canonical in CANONICAL_CLASSES:
        (split_dir / canonical).mkdir(exist_ok=True)
    
    # Move files from alias directories to canonical directories
    for canonical, aliases in CLASS_ALIASES.items():
        target_dir = split_dir / canonical
        for alias in aliases:
            alias_dir = split_dir / alias
            if alias_dir.exists() and alias_dir.resolve() != target_dir.resolve():
                for item in alias_dir.iterdir():
                    destination = target_dir / item.name
                    # Handle duplicates
                    if destination.exists():
                        base = destination.stem
                        suffix = destination.suffix
                        idx = 1
                        while destination.exists():
                            destination = target_dir / f'{base}_{idx}{suffix}'
                            idx += 1
                    item.rename(destination)
                alias_dir.rmdir()


def count_images(split_dir: Path) -> Dict[str, int]:
    """Count images in each class"""
    counts = {}
    for cls in CANONICAL_CLASSES:
        cls_dir = split_dir / cls
        if cls_dir.exists():
            counts[cls] = len([p for p in cls_dir.glob('*') if p.is_file()])
        else:
            counts[cls] = 0
    return counts


def validate_dataset() -> None:
    """Validate and organize dataset structure"""
    print('\n' + '='*60)
    print('DATASET VALIDATION')
    print('='*60)
    
    for split in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        ensure_dataset_layout(split)
    
    train_counts = count_images(TRAIN_DIR)
    val_counts = count_images(VAL_DIR)
    test_counts = count_images(TEST_DIR)
    
    print(f'Train: {train_counts}')
    print(f'Val  : {val_counts}')
    print(f'Test : {test_counts}')
    
    # Validate minimum requirements
    if sum(train_counts.values()) < 100:
        raise RuntimeError('⚠ Training set too small (minimum 100 images required)')
    if sum(val_counts.values()) < 20:
        raise RuntimeError('⚠ Validation set too small (minimum 20 images required)')
    if sum(test_counts.values()) < 20:
        raise RuntimeError('⚠ Test set too small (minimum 20 images required)')
    
    # Check class balance
    for split_name, counts in [('Train', train_counts), ('Val', val_counts)]:
        if counts['with_mask'] > 0 and counts['without_mask'] > 0:
            ratio = max(counts['with_mask'], counts['without_mask']) / min(counts['with_mask'], counts['without_mask'])
            if ratio > 3:
                print(f'⚠ {split_name} set is imbalanced (ratio: {ratio:.1f}:1)')
    
    print('='*60 + '\n')


# ============================================================
# DATA GENERATORS
# ============================================================

def create_generators() -> Tuple:
    """Create optimized data generators for training"""
    
    # REDUCED augmentation to prevent overfitting
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,           # Reduced from 25
        width_shift_range=0.1,       # Reduced from 0.2
        height_shift_range=0.1,      # Reduced from 0.2
        horizontal_flip=True,
        zoom_range=0.1,              # Reduced from 0.2
        brightness_range=[0.8, 1.2], # Reduced range
        fill_mode='nearest',
        # Removed shear to prevent unrealistic distortions
    )
    
    # No augmentation for validation/test
    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    
    common = {
        'target_size': (IMG_SIZE, IMG_SIZE),
        'classes': CANONICAL_CLASSES,
        'class_mode': 'binary',
        'batch_size': BATCH_SIZE,
        'seed': RANDOM_SEED,
    }
    
    train_generator = train_datagen.flow_from_directory(
        str(TRAIN_DIR),
        shuffle=True,
        **common,
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        str(VAL_DIR),
        shuffle=False,
        **common,
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        str(TEST_DIR),
        shuffle=False,
        **common,
    )
    
    return train_generator, val_generator, test_generator


def compute_class_weights(train_generator) -> Dict[int, float]:
    """Compute balanced class weights"""
    counts = np.bincount(train_generator.classes)
    total = counts.sum()
    n_classes = len(counts)
    
    weights = {}
    for class_idx, class_count in enumerate(counts):
        if class_count > 0:
            # Balanced weighting formula
            weights[class_idx] = float(total / (n_classes * class_count))
        else:
            weights[class_idx] = 1.0
    
    return weights


# ============================================================
# MODEL ARCHITECTURE
# ============================================================

def build_production_model() -> Tuple[keras.Model, keras.Model]:
    """
    Build production-ready model using MobileNetV2
    
    MobileNetV2 advantages:
    - Optimized for mobile/edge devices
    - Better feature extraction for faces
    - Faster inference
    - Smaller model size
    """
    
    # Load pretrained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        alpha=1.0  # Width multiplier
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build classifier head
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input_image')
    
    # Base model feature extraction
    x = base_model(inputs, training=False)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    
    # Batch normalization for stability
    x = layers.BatchNormalization(name='bn_1')(x)
    
    # First dense layer with regularization
    x = layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(L2_WEIGHT),
        name='dense_1'
    )(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.Dropout(DROPOUT_RATE, name='dropout_1')(x)
    
    # Second dense layer
    x = layers.Dense(
        128,
        activation='relu',
        kernel_regularizer=keras.regularizers.l2(L2_WEIGHT),
        name='dense_2'
    )(x)
    x = layers.Dropout(DROPOUT_RATE * 0.5, name='dropout_2')(x)
    
    # Output layer (float32 for numerical stability)
    outputs = layers.Dense(
        1,
        activation='sigmoid',
        dtype='float32',
        name='output'
    )(x)
    
    model = keras.Model(inputs, outputs, name='MaskDetector_Production')
    
    return model, base_model


# ============================================================
# TRAINING CALLBACKS
# ============================================================

def create_callbacks(phase: str) -> list:
    """Create training callbacks for each phase"""
    
    callbacks = []
    
    # Model checkpoint - save best model
    checkpoint = ModelCheckpoint(
        str(MODEL_H5_PATH),
        monitor='val_accuracy',  # Changed from val_loss
        save_best_only=True,
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint)
    
    # Early stopping - prevent overfitting
    if phase == 'phase1':
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=8,  # Increased patience
            restore_best_weights=True,
            verbose=1
        )
    else:  # phase2
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    callbacks.append(early_stop)
    
    # Learning rate reduction on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=MIN_LR,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    return callbacks


# ============================================================
# EVALUATION
# ============================================================

def compute_optimal_threshold(model, val_generator) -> float:
    """Find optimal classification threshold using validation set"""
    
    print('\nComputing optimal threshold...')
    val_generator.reset()
    
    # Get predictions
    y_true = val_generator.classes
    y_pred_proba = model.predict(val_generator, verbose=0).flatten()
    
    # Try different thresholds
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f'✓ Optimal threshold: {best_threshold:.3f} (F1: {best_f1:.4f})')
    return float(best_threshold)


def evaluate_model(model, test_generator, threshold: float) -> Dict:
    """Comprehensive model evaluation"""
    
    print('\n' + '='*60)
    print('MODEL EVALUATION')
    print('='*60)
    
    test_generator.reset()
    
    # Get predictions
    y_true = test_generator.classes
    y_pred_proba = model.predict(test_generator, verbose=0).flatten()
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics for each class
    results = {}
    
    for idx, class_name in enumerate(CANONICAL_CLASSES):
        mask = (y_true == idx)
        if mask.sum() == 0:
            continue
            
        tp = np.sum((y_pred == idx) & (y_true == idx))
        fp = np.sum((y_pred == idx) & (y_true != idx))
        fn = np.sum((y_pred != idx) & (y_true == idx))
        tn = np.sum((y_pred != idx) & (y_true != idx))
        
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        accuracy = (tp + tn) / len(y_true)
        
        results[class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'support': int(mask.sum())
        }
        
        print(f'\n{class_name.upper()}:')
        print(f'  Precision: {precision:.4f}')
        print(f'  Recall:    {recall:.4f}')
        print(f'  F1-Score:  {f1:.4f}')
        print(f'  Support:   {mask.sum()}')
    
    # Overall accuracy
    overall_accuracy = np.mean(y_pred == y_true)
    results['overall_accuracy'] = float(overall_accuracy)
    
    print(f'\nOVERALL ACCURACY: {overall_accuracy:.4f}')
    print('='*60 + '\n')
    
    return results


# ============================================================
# VISUALIZATION
# ============================================================

def plot_training_history(history1, history2):
    """Plot training history"""
    
    # Combine histories
    metrics = {
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
        'loss': history1.history['loss'] + history2.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(metrics['accuracy'], label='Training', linewidth=2)
    axes[0].plot(metrics['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(metrics['loss'], label='Training', linewidth=2)
    axes[1].plot(metrics['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches='tight')
    print(f'✓ Training plot saved: {PLOT_PATH}')


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    """Main training pipeline"""
    
    print('\n' + '='*60)
    print('PRODUCTION MASK DETECTION MODEL TRAINING')
    print('='*60)
    print(f'TensorFlow version: {tf.__version__}')
    print(f'Keras version: {keras.__version__}')
    
    # Setup
    setup_runtime()
    validate_dataset()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create data generators
    train_gen, val_gen, test_gen = create_generators()
    print(f'\n✓ Class indices: {train_gen.class_indices}')
    
    # Compute class weights
    class_weights = compute_class_weights(train_gen)
    print(f'✓ Class weights: {class_weights}')
    
    # Build model
    print('\nBuilding MobileNetV2-based model...')
    model, base_model = build_production_model()
    print(model.summary())
    
    # ============================================================
    # PHASE 1: Train classifier head
    # ============================================================
    
    print('\n' + '='*60)
    print('PHASE 1: Training Classifier Head')
    print('='*60)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy'],
    )
    
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE1,
        callbacks=create_callbacks('phase1'),
        class_weight=class_weights,
        verbose=1,
    )
    
    # ============================================================
    # PHASE 2: Fine-tune entire model
    # ============================================================
    
    print('\n' + '='*60)
    print('PHASE 2: Fine-Tuning Entire Model')
    print('='*60)
    
    # Unfreeze base model layers
    base_model.trainable = True
    
    # Fine-tune only the last 30 layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    print(f'✓ Trainable layers: {sum([1 for layer in model.layers if layer.trainable])}')
    
    # Compile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss=keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=['accuracy'],
    )
    
    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE2,
        callbacks=create_callbacks('phase2'),
        class_weight=class_weights,
        verbose=1,
    )
    
    # ============================================================
    # EVALUATION
    # ============================================================
    
    # Load best model
    model = keras.models.load_model(MODEL_H5_PATH)
    print(f'\n✓ Loaded best model from: {MODEL_H5_PATH}')
    
    # Find optimal threshold
    optimal_threshold = compute_optimal_threshold(model, val_gen)
    
    # Evaluate on test set
    test_results = evaluate_model(model, test_gen, optimal_threshold)
    
    # ============================================================
    # SAVE METADATA
    # ============================================================
    
    metadata = {
        'model_architecture': 'MobileNetV2',
        'model_path': str(MODEL_H5_PATH),
        'img_size': IMG_SIZE,
        'class_indices': {k: int(v) for k, v in train_gen.class_indices.items()},
        'optimal_threshold': optimal_threshold,
        'train_counts': count_images(TRAIN_DIR),
        'val_counts': count_images(VAL_DIR),
        'test_counts': count_images(TEST_DIR),
        'test_results': test_results,
        'training_config': {
            'batch_size': BATCH_SIZE,
            'epochs_phase1': EPOCHS_PHASE1,
            'epochs_phase2': EPOCHS_PHASE2,
            'initial_lr': INITIAL_LR,
            'fine_tune_lr': FINE_TUNE_LR,
            'dropout_rate': DROPOUT_RATE,
            'l2_weight': L2_WEIGHT,
            'label_smoothing': LABEL_SMOOTHING,
        }
    }
    
    with METADATA_PATH.open('w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f'✓ Metadata saved: {METADATA_PATH}')
    
    # Plot training history
    plot_training_history(history1, history2)
    
    print('\n' + '='*60)
    print('TRAINING COMPLETED SUCCESSFULLY')
    print('='*60)
    print(f'Model saved: {MODEL_H5_PATH}')
    print(f'Overall test accuracy: {test_results["overall_accuracy"]:.2%}')
    print('='*60 + '\n')


if __name__ == '__main__':
    main()