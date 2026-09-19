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

TRAIN_CSV = OUTPUT_DIR / "train_split.csv"
VAL_CSV = OUTPUT_DIR / "val_split.csv"
CLASS_MAP_JSON = OUTPUT_DIR / "class_mapping.json"
CLASS_WEIGHTS_JSON = OUTPUT_DIR / "class_weights.json"

PREV_CHECKPOINT = MODELS_DIR / "feature_extraction_best.keras"
FINAL_MODEL_PATH = MODELS_DIR / "disease_classifier.keras"
HISTORY_JSON = OUTPUT_DIR / "phase10_history.json"
PLOT_PATH = OUTPUT_DIR / "phase10_fine_tuning_curves.png"

def run_fine_tuning(epochs=10, fine_tune_layers=40, fine_tune_lr=1e-5):
    print("=" * 60)
    print("          PHASE 10: FINE-TUNING EFFICIENTNETV2B0          ")
    print("=" * 60)

    from phase4_pipeline import create_tf_dataset

    if not PREV_CHECKPOINT.exists():
        print(f"[ERROR] Previous checkpoint not found at {PREV_CHECKPOINT.resolve()}. Run Phase 9 first.")
        sys.exit(1)

    # 1. Load Data Pipelines & Class Mapping/Weights
    with open(CLASS_MAP_JSON, 'r') as f:
        class_map = json.load(f)
    with open(CLASS_WEIGHTS_JSON, 'r') as f:
        raw_class_weights = json.load(f)
        
    class_weights = {int(k): float(v) for k, v in raw_class_weights.items()}

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    train_ds = create_tf_dataset(train_df, class_map, batch_size=32, is_training=True)
    val_ds = create_tf_dataset(val_df, class_map, batch_size=32, is_training=False)

    # 2. Load Phase 9 Best Checkpoint Model
    print(f"\n1. LOADING PHASE 9 BEST MODEL CHECKPOINT: {PREV_CHECKPOINT.name}...")
    model = tf.keras.models.load_model(PREV_CHECKPOINT)

    # Find EfficientNetV2B0 base model layer
    base_model = None
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        print("[ERROR] Base model EfficientNetV2B0 layer not found inside loaded model!")
        sys.exit(1)

    print(f"   Found Base Model Layer: {base_model.name} (Total Base Layers: {len(base_model.layers)})")

    # 3. Unfreeze Later Layers & Handle BatchNormalization Carefully
    print(f"\n2. UNFREEZING TOP {fine_tune_layers} LAYERS OF EFFICIENTNETV2B0...")
    base_model.trainable = True

    # Freeze all early layers up to the last `fine_tune_layers`
    freeze_until = len(base_model.layers) - fine_tune_layers
    for idx, layer in enumerate(base_model.layers):
        if idx < freeze_until:
            layer.trainable = False
        else:
            layer.trainable = True
            
        # CRITICAL BATCHNORMALIZATION SAFETY:
        # Keep BatchNormalization layers frozen during fine-tuning so moving statistics are not corrupted!
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # Print summary of trainable layers
    trainable_base_layers = [l.name for l in base_model.layers if l.trainable]
    print(f"   Frozen Base Layers Count    : {freeze_until}")
    print(f"   Unfrozen Base Layers Count  : {len(trainable_base_layers)}")

    # 4. Recompile Model with Reduced Learning Rate
    print(f"\n3. RECOMPILING MODEL WITH SMALL LEARNING RATE (LR = {fine_tune_lr})...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=fine_tune_lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )

    trainable_count = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    non_trainable_count = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    print(f"   Trainable Parameters        : {trainable_count:,}")
    print(f"   Non-Trainable Parameters    : {non_trainable_count:,}")

    # 5. Callbacks Setup
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(FINAL_MODEL_PATH),
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
            min_lr=1e-7,
            verbose=1
        )
    ]

    # 6. Execute Fine-Tuning
    print("\n4. STARTING FINE-TUNING TRAINING...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # 7. Save History Data
    hist_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
    with open(HISTORY_JSON, 'w') as f:
        json.dump(hist_dict, f, indent=4)
    print(f"\n5. SAVED FINE-TUNING HISTORY TO: {HISTORY_JSON.resolve()}")

    # 8. Plot Fine-Tuning Curves
    print("\n6. PLOTTING FINE-TUNING CURVES...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(hist_dict['accuracy']) + 1)

    axes[0].plot(epochs_range, hist_dict['accuracy'], 'o-', label='Fine-Tune Train Accuracy', color='blue')
    axes[0].plot(epochs_range, hist_dict['val_accuracy'], 's-', label='Fine-Tune Val Accuracy', color='green')
    axes[0].set_title('Phase 10: Fine-Tuning Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs_range, hist_dict['loss'], 'o-', label='Fine-Tune Train Loss', color='red')
    axes[1].plot(epochs_range, hist_dict['val_loss'], 's-', label='Fine-Tune Val Loss', color='orange')
    axes[1].set_title('Phase 10: Fine-Tuning Loss', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"   Saved fine-tuning curves plot to: {PLOT_PATH.resolve()}")

    print("\n" + "=" * 60)
    print("                PHASE 10 FINE-TUNING COMPLETE               ")
    print("=" * 60)
    print(f" Final Saved Classifier: {FINAL_MODEL_PATH.resolve()}")
    print(" Ready for Phase 11 (Final Test Set Evaluation).")
    print("=" * 60)

if __name__ == "__main__":
    run_fine_tuning(epochs=10, fine_tune_layers=40, fine_tune_lr=1e-5)
