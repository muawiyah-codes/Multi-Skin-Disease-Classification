import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import tensorflow as tf

# Set seed
tf.random.set_seed(42)
np.random.seed(42)

OUTPUT_DIR = Path("outputs")
MODELS_DIR = Path("models")

TEST_CSV = OUTPUT_DIR / "test_split.csv"
CLASS_MAP_JSON = OUTPUT_DIR / "class_mapping.json"
FINAL_MODEL_PATH = MODELS_DIR / "disease_classifier.keras"

CONF_MATRIX_PLOT = OUTPUT_DIR / "phase11_confusion_matrix.png"
REPORT_CSV = OUTPUT_DIR / "phase11_classification_report.csv"
METRICS_JSON = OUTPUT_DIR / "phase11_overall_metrics.json"

def evaluate_on_test_set():
    print("=" * 60)
    print("      PHASE 11: UNTOUCHED TEST SET FINAL EVALUATION       ")
    print("=" * 60)

    from phase4_pipeline import create_tf_dataset

    if not FINAL_MODEL_PATH.exists():
        print(f"[ERROR] Final trained model not found at {FINAL_MODEL_PATH.resolve()}. Run Phase 10 first.")
        sys.exit(1)

    # 1. Load Model & Specs
    print(f"\n1. LOADING FINAL TRAINED MODEL: {FINAL_MODEL_PATH.name}...")
    model = tf.keras.models.load_model(FINAL_MODEL_PATH)

    with open(CLASS_MAP_JSON, 'r') as f:
        class_map = json.load(f)
        
    inv_class_map = {idx: name for name, idx in class_map.items()}
    class_names = [inv_class_map[i] for i in range(len(class_map))]

    test_df = pd.read_csv(TEST_CSV)
    print(f"   Test Set Size: {len(test_df)} images")

    # Create un-shuffled test dataset pipeline
    test_ds = create_tf_dataset(test_df, class_map, batch_size=32, is_training=False)

    # 2. Run Inference on Test Dataset
    print("\n2. RUNNING INFERENCE ON TEST DATASET...")
    y_true = []
    y_pred_probs = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_pred_probs.append(probs)
        y_true.extend(labels.numpy())

    y_pred_probs = np.vstack(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.array(y_true)

    # 3. Overall Metrics Calculation
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    print("\n" + "=" * 60)
    print("                OVERALL TEST PERFORMANCE SUMMARY            ")
    print("=" * 60)
    print(f" Test Accuracy    : {acc*100:.2f}%")
    print(f" Macro Precision  : {macro_p*100:.2f}%")
    print(f" Macro Recall     : {macro_r*100:.2f}%")
    print(f" Macro F1-Score   : {macro_f1*100:.2f}%")
    print(f" Weighted F1-Score: {weighted_f1*100:.2f}%")
    print("=" * 60)

    # Save overall metrics
    overall_metrics = {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1)
    }
    with open(METRICS_JSON, 'w') as f:
        json.dump(overall_metrics, f, indent=4)

    # 4. Detailed Per-Class Classification Report Table
    print("\n3. PER-CLASS CLASSIFICATION METRICS TABLE:")
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    df_report.to_csv(REPORT_CSV)
    print(df_report.round(4).to_string())

    # 5. Confusion Matrix Visualization
    print("\n4. GENERATING CONFUSION MATRIX HEATMAP...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title("Phase 11: Test Set Confusion Matrix", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CONF_MATRIX_PLOT, dpi=150)
    plt.close()
    print(f"   Saved Confusion Matrix plot to: {CONF_MATRIX_PLOT.resolve()}")

    print("\n" + "=" * 60)
    print("             PHASE 11 TEST EVALUATION COMPLETE              ")
    print("=" * 60)
    print(f" Classification Report Saved: {REPORT_CSV.resolve()}")
    print(" Ready for Phase 12 (Error Analysis).")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_on_test_set()
