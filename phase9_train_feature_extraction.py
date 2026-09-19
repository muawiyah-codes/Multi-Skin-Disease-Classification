import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)

OUTPUT_DIR = Path("outputs")
MODELS_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

TRAIN_CSV = OUTPUT_DIR / "train_split.csv"
VAL_CSV = OUTPUT_DIR / "val_split.csv"
CLASS_MAP_JSON = OUTPUT_DIR / "class_mapping.json"
CLASS_WEIGHTS_JSON = OUTPUT_DIR / "class_weights.json"
CHECKPOINT_PATH = MODELS_DIR / "feature_extraction_best.keras"
HISTORY_JSON = OUTPUT_DIR / "phase9_history.json"
PLOT_PATH = OUTPUT_DIR / "phase9_training_curves.png"

def run_feature_extraction_training(epochs=10):
    print("=" * 60)
    print("     PHASE 9: FEATURE EXTRACTION TRAINING (BASE FROZEN)    ")
    print("=" * 60)

    from phase4_pipeline import create_tf_dataset
    from phase7_model import build_transfer_learning_model

    # 1. Load Specs & Pipelines
    with open(CLASS_MAP_JSON, 'r') as f:
        class_map = json.load(f)
    with open(CLASS_WEIGHTS_JSON, 'r') as f:
        raw_class_weights = json.load(f)
        
    class_weights = {int(k): float(v) for k, v in raw_class_weights.items()}
    num_classes = len(class_map)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    print("\n1. CREATING TF.DATA INPUT PIPELINES...")
    train_ds = create_tf_dataset(train_df, class_map, batch_size=32, is_training=True)
    val_ds = create_tf_dataset(val_df, class_map, batch_size=32, is_training=False)

    # 2. Build Model
    print("\n2. ASSEMBLING MODEL (EFFICIENTNETV2B0 FROZEN)...")
    model, base_model = build_transfer_learning_model(num_classes=num_classes)
    
    # Verify base model is frozen
    base_model.trainable = False

    # 3. Compile Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )
    print(f"   Optimizer   : Adam (LR = 1e-3)")
    print(f"   Loss        : SparseCategoricalCrossentropy")
    print(f"   Metrics     : ['accuracy']")

    # 4. Setup Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CHECKPOINT_PATH),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("\n3. STARTING FEATURE EXTRACTION TRAINING...")
    print(f"   Epochs: {epochs}")
    print(f"   Class Weights Active: {class_weights}")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # 5. Save History Data
    hist_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
    with open(HISTORY_JSON, 'w') as f:
        json.dump(hist_dict, f, indent=4)
    print(f"\n4. SAVED TRAINING HISTORY TO: {HISTORY_JSON.resolve()}")

    # 6. Plot Training & Validation Curves
    print("\n5. PLOTTING TRAINING & VALIDATION CURVES...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_range = range(1, len(hist_dict['accuracy']) + 1)
    
    # Accuracy Plot
    axes[0].plot(epochs_range, hist_dict['accuracy'], 'o-', label='Training Accuracy', color='blue')
    axes[0].plot(epochs_range, hist_dict['val_accuracy'], 's-', label='Validation Accuracy', color='green')
    axes[0].set_title('Phase 9: Feature Extraction Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss Plot
    axes[1].plot(epochs_range, hist_dict['loss'], 'o-', label='Training Loss', color='red')
    axes[1].plot(epochs_range, hist_dict['val_loss'], 's-', label='Validation Loss', color='orange')
    axes[1].set_title('Phase 9: Feature Extraction Loss', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"   Saved training curves plot to: {PLOT_PATH.resolve()}")

    print("\n" + "=" * 60)
    print("          PHASE 9 FEATURE EXTRACTION COMPLETE              ")
    print("=" * 60)
    print(f" Best Checkpoint Saved: {CHECKPOINT_PATH.resolve()}")
    print(" Ready for Phase 10 (Fine-Tuning).")
    print("=" * 60)

if __name__ == "__main__":
    run_feature_extraction_training(epochs=10)
