import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

OUTPUT_DIR = Path("outputs")
MODELS_DIR = Path("models")
VALIDATOR_DATA_DIR = Path("data/validator_dataset")
VALIDATOR_DATA_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = OUTPUT_DIR / "train_split.csv"
VAL_CSV = OUTPUT_DIR / "val_split.csv"
VALIDATOR_MODEL_PATH = MODELS_DIR / "image_validator.keras"
VALIDATOR_REPORT_CSV = OUTPUT_DIR / "phase14_validator_report.csv"
VALIDATOR_CM_PLOT = OUTPUT_DIR / "phase14_validator_confusion_matrix.png"

def create_synthetic_ood_images(output_dir, num_samples=500):
    invalid_dir = output_dir / "INVALID"
    invalid_dir.mkdir(exist_ok=True)
    
    existing_files = list(invalid_dir.glob("*.jpg"))
    if len(existing_files) >= num_samples:
        return [str(f) for f in existing_files]

    generated_paths = []
    for i in range(num_samples):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        sample_type = i % 5
        
        if sample_type == 0:
            skin_color = (int(np.random.randint(140, 220)), int(np.random.randint(180, 240)), int(np.random.randint(200, 255)))
            cv2.ellipse(img, (112, 112), (70, 90), 0, 0, 360, skin_color, -1)
            cv2.circle(img, (90, 90), 12, (50, 50, 50), -1)
            cv2.circle(img, (134, 90), 12, (50, 50, 50), -1)
            cv2.ellipse(img, (112, 140), (25, 10), 0, 0, 180, (40, 40, 180), 3)
            
        elif sample_type == 1:
            img.fill(245)
            for line_y in range(30, 200, 20):
                cv2.line(img, (20, line_y), (int(np.random.randint(120, 200)), line_y), (30, 30, 30), 2)
                
        elif sample_type == 2:
            img[:] = (int(np.random.randint(50, 150)), int(np.random.randint(50, 150)), int(np.random.randint(50, 150)))
            cv2.rectangle(img, (30, 100), (194, 170), (200, 30, 30), -1)
            cv2.circle(img, (65, 170), 20, (20, 20, 20), -1)
            cv2.circle(img, (159, 170), 20, (20, 20, 20), -1)
            
        elif sample_type == 3:
            img[:112] = (235, 180, 100)
            img[112:] = (50, 140, 60)
            cv2.circle(img, (180, 40), 25, (100, 230, 255), -1)
            
        else:
            noise = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            img = cv2.addWeighted(img, 0.2, noise, 0.8, 0)
            
        fpath = invalid_dir / f"invalid_ood_{i:04d}.jpg"
        cv2.imwrite(str(fpath), img)
        generated_paths.append(str(fpath))
        
    return generated_paths

def build_validator_dataset():
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    
    valid_paths = list(pd.concat([train_df['filepath'], val_df['filepath']]).sample(500, random_state=42).values)
    invalid_paths = create_synthetic_ood_images(VALIDATOR_DATA_DIR, num_samples=500)
    
    records = []
    for p in valid_paths:
        records.append({"filepath": p, "label": 1, "class_name": "VALID"})
    for p in invalid_paths:
        records.append({"filepath": p, "label": 0, "class_name": "INVALID"})
        
    df_val_dataset = pd.DataFrame(records).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df_val_dataset

def build_lightweight_validator_model():
    inputs = tf.keras.Input(shape=(224, 224, 3), name="validator_input")
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False
    
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="validator_output")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="SkinDomain_Validator")
    return model

def run_validator_pipeline():
    print("=" * 60)
    print("    PHASE 14: INPUT VALIDATOR / OOD PROTECTION SYSTEM    ")
    print("=" * 60)

    df_validator = build_validator_dataset()
    print(f"\n1. ASSEMBLING VALIDATOR DATASET (LEAKAGE SAFE): {len(df_validator)} images")

    train_idx = int(0.8 * len(df_validator))
    df_vtrain = df_validator.iloc[:train_idx]
    df_vtest = df_validator.iloc[train_idx:]

    def parse_val_image(path, label):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [224, 224])
        return tf.cast(img, tf.float32), label

    train_ds = tf.data.Dataset.from_tensor_slices((df_vtrain['filepath'].values, df_vtrain['label'].values))
    train_ds = train_ds.map(parse_val_image).batch(16).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((df_vtest['filepath'].values, df_vtest['label'].values))
    test_ds = test_ds.map(parse_val_image).batch(16).prefetch(tf.data.AUTOTUNE)

    print("\n2. BUILDING & TRAINING LIGHTWEIGHT BINARY VALIDATOR MODEL (MobileNetV2)...")
    if VALIDATOR_MODEL_PATH.exists():
        print(f"   Loading existing trained validator model: {VALIDATOR_MODEL_PATH.name}")
        validator_model = tf.keras.models.load_model(VALIDATOR_MODEL_PATH)
    else:
        validator_model = build_lightweight_validator_model()
        validator_model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )
        validator_model.fit(train_ds, validation_data=test_ds, epochs=5, verbose=1)
        validator_model.save(VALIDATOR_MODEL_PATH)

    print("\n3. EVALUATING VALIDATOR PERFORMANCE ON UNSEEN OOD TEST POOL...")
    y_true = []
    y_pred_probs = []
    
    for imgs, lbls in test_ds:
        preds = validator_model.predict(imgs, verbose=0)
        y_pred_probs.extend(preds.flatten())
        y_true.extend(lbls.numpy())

    y_pred_probs = np.array(y_pred_probs)
    y_pred = (y_pred_probs >= 0.5).astype(int)
    y_true = np.array(y_true)

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    print("\n" + "=" * 60)
    print("             IMAGE VALIDATOR PERFORMANCE SUMMARY           ")
    print("=" * 60)
    print(f" Validator Accuracy  : {acc*100:.2f}%")
    print(f" Validator Precision : {p*100:.2f}%")
    print(f" Validator Recall    : {r*100:.2f}%")
    print(f" Validator F1-Score  : {f1*100:.2f}%")
    print("=" * 60)

    report_dict = classification_report(y_true, y_pred, target_names=['INVALID', 'VALID'], output_dict=True)
    pd.DataFrame(report_dict).transpose().to_csv(VALIDATOR_REPORT_CSV)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['INVALID', 'VALID'], yticklabels=['INVALID', 'VALID'])
    plt.title("Phase 14: Image Validator Confusion Matrix", fontsize=12, fontweight='bold')
    plt.xlabel("Predicted Status")
    plt.ylabel("True Status")
    plt.tight_layout()
    plt.savefig(VALIDATOR_CM_PLOT, dpi=150)
    plt.close()
    print(f"   Saved Validator Confusion Matrix to: {VALIDATOR_CM_PLOT.resolve()}")

    print("\n" + "=" * 60)
    print("          PHASE 14 OOD PROTECTION SYSTEM COMPLETE           ")
    print("=" * 60)
    print(f" Binary Validator Model Saved: {VALIDATOR_MODEL_PATH.resolve()}")
    print(" Ready for Phase 15 (Uncertainty Rejection System).")
    print("=" * 60)

if __name__ == "__main__":
    run_validator_pipeline()
