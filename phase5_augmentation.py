import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Set seeds
tf.random.set_seed(42)
np.random.seed(42)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def build_data_augmentation_layer():
    """
    Builds GPU-accelerated Keras Data Augmentation Pipeline.
    Applied strictly ONLY during model training.
    """
    data_augmentation = tf.keras.Sequential([
        # Medical lesions are orientation invariant (flipping top-down or left-right is clinically safe)
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=42, name="aug_random_flip"),
        
        # Mild rotation (+/- 15% of 360 degrees = +/- 54 degrees)
        tf.keras.layers.RandomRotation(0.15, fill_mode="reflect", seed=42, name="aug_random_rotation"),
        
        # Mild zoom (+/- 10%)
        tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), fill_mode="reflect", seed=42, name="aug_random_zoom"),
        
        # Slight lighting/contrast adjustment (+/- 10%) to simulate different camera lighting
        tf.keras.layers.RandomContrast(0.10, seed=42, name="aug_random_contrast")
    ], name="data_augmentation_pipeline")
    
    return data_augmentation

def test_augmentation_visuals():
    print("=" * 60)
    print("         PHASE 5: DATA AUGMENTATION PIPELINE SETUP       ")
    print("=" * 60)

    from phase4_pipeline import create_tf_dataset, TRAIN_CSV, CLASS_MAP_JSON
    import json
    import pandas as pd

    if not TRAIN_CSV.exists() or not CLASS_MAP_JSON.exists():
        print("[ERROR] Required Phase 3/4 files missing. Please run Phase 3 & 4 first.")
        sys.exit(1)

    with open(CLASS_MAP_JSON, 'r') as f:
        class_map = json.load(f)
    inv_class_map = {idx: name for name, idx in class_map.items()}

    train_df = pd.read_csv(TRAIN_CSV)
    train_ds = create_tf_dataset(train_df, class_map, batch_size=32, is_training=False)

    aug_layer = build_data_augmentation_layer()
    
    print("\n1. AUGMENTATION LAYERS SUMMARY:")
    aug_layer.summary()

    # Extract 1 sample image from train dataset
    print("\n2. APPLYING AUGMENTATION TO SAMPLE LESION IMAGE...")
    for images, labels in train_ds.take(1):
        sample_img = images[0] # (224, 224, 3)
        sample_lbl = labels[0].numpy()
        sample_name = inv_class_map[sample_lbl]
        break

    # Pass 1 sample image through augmentation layer 6 times
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Phase 5: Data Augmentation Variations for Class '{sample_name}'", fontsize=14, fontweight='bold')

    # Slot 0: Original Image
    ax_orig = axes[0, 0]
    ax_orig.imshow(sample_img.numpy().astype("uint8"))
    ax_orig.set_title("ORIGINAL IMAGE\n(No Augmentation)", fontsize=10, fontweight='bold', color='darkblue')
    ax_orig.axis('off')

    # Slot 1 to 6: 6 Randomly Augmented Versions
    for i in range(1, 7):
        row = i // 4
        col = i % 4
        ax = axes[row, col]
        
        # Expand batch dim (1, 224, 224, 3) and augment
        augmented_batch = aug_layer(tf.expand_dims(sample_img, 0), training=True)
        augmented_img = augmented_batch[0].numpy().astype("uint8")
        
        ax.imshow(augmented_img)
        ax.set_title(f"Augmented Variation #{i}", fontsize=10)
        ax.axis('off')

    # Hide unused grid subplot (slot 7)
    axes[1, 3].axis('off')

    plt.tight_layout()
    aug_out_path = OUTPUT_DIR / "phase5_augmented_samples.png"
    plt.savefig(aug_out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved augmentation visual grid to: {aug_out_path.resolve()}")

    print("\n" + "=" * 60)
    print("                PHASE 5 AUGMENTATION VERIFICATION          ")
    print("=" * 60)
    print(" Data augmentation layer compiled successfully.")
    print(" Applies strictly to Training data ONLY during model.fit().")
    print(" Validation & Test datasets remain UN-AUGMENTED for clean evaluation.")
    print(" Ready for Phase 6 (EfficientNetV2 Preprocessing).")
    print("=" * 60)

if __name__ == "__main__":
    test_augmentation_visuals()
