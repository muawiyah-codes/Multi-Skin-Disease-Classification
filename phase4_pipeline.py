import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf

# Set random seed
tf.random.set_seed(42)
np.random.seed(42)

OUTPUT_DIR = Path("outputs")
TRAIN_CSV = OUTPUT_DIR / "train_split.csv"
VAL_CSV = OUTPUT_DIR / "val_split.csv"
TEST_CSV = OUTPUT_DIR / "test_split.csv"
CLASS_MAP_JSON = OUTPUT_DIR / "class_mapping.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def parse_image(filepath, label):
    """Load, decode JPEG/PNG, convert channels to RGB, and resize to 224x224."""
    image_raw = tf.io.read_file(filepath)
    image = tf.io.decode_image(image_raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE, method='bilinear')
    # Cast to float32 [0.0, 255.0] range
    image = tf.cast(image, tf.float32)
    return image, label

def create_tf_dataset(df, class_map, batch_size=32, is_training=False):
    filepaths = df['filepath'].values
    labels = [class_map[c] for c in df['class'].values]
    
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(df), seed=42)
        
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

def run_pipeline_setup():
    print("=" * 60)
    print("         PHASE 4: TENSORFLOW DATA PIPELINE SETUP         ")
    print("=" * 60)

    if not TRAIN_CSV.exists():
        print(f"[ERROR] Train CSV not found at {TRAIN_CSV.resolve()}. Please run Phase 3 first.")
        sys.exit(1)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # 1. Create & Save Class Mapping
    class_names = sorted(train_df['class'].unique())
    class_map = {name: idx for idx, name in enumerate(class_names)}
    inv_class_map = {idx: name for name, idx in class_map.items()}

    with open(CLASS_MAP_JSON, 'w') as f:
        json.dump(class_map, f, indent=4)
        
    print(f"\n1. CLASS MAPPING DICTIONARY (Saved to {CLASS_MAP_JSON.name}):")
    for name, idx in class_map.items():
        print(f"   Label {idx} -> Class '{name}'")

    # 2. Build Datasets
    print("\n2. CREATING TF.DATA PIPELINES (Batch Size = 32, Target Size = 224x224x3)...")
    train_ds = create_tf_dataset(train_df, class_map, batch_size=BATCH_SIZE, is_training=True)
    val_ds = create_tf_dataset(val_df, class_map, batch_size=BATCH_SIZE, is_training=False)
    test_ds = create_tf_dataset(test_df, class_map, batch_size=BATCH_SIZE, is_training=False)

    print(f"   Train batches count : {len(train_ds)}")
    print(f"   Val batches count   : {len(val_ds)}")
    print(f"   Test batches count  : {len(test_ds)}")

    # 3. Batch Inspection
    print("\n3. INSPECTING 1 BATCH FROM TRAIN PIPELINE:")
    for images, labels in train_ds.take(1):
        print(f"   Image Batch Shape : {images.shape} (Batch, Height, Width, Channels)")
        print(f"   Label Batch Shape : {labels.shape}")
        print(f"   Pixel Range       : Min = {tf.reduce_min(images):.1f}, Max = {tf.reduce_max(images):.1f}")
        print(f"   Data Type         : {images.dtype}")
        print(f"   First 5 Labels    : {labels.numpy()[:5]}")
        print(f"   First 5 Names     : {[inv_class_map[lbl] for lbl in labels.numpy()[:5]]}")

    print("\n" + "=" * 60)
    print("                PHASE 4 PIPELINE VERIFICATION              ")
    print("=" * 60)
    print(f" TensorFlow pipeline verified successfully.")
    print(f" Efficient asynchronous loading (AUTOTUNE + Prefetch) active.")
    print(" Ready for Phase 5 (Data Augmentation).")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline_setup()
