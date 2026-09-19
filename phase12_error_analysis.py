import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

OUTPUT_DIR = Path("outputs")
MODELS_DIR = Path("models")

TEST_CSV = OUTPUT_DIR / "test_split.csv"
CLASS_MAP_JSON = OUTPUT_DIR / "class_mapping.json"
FINAL_MODEL_PATH = MODELS_DIR / "disease_classifier.keras"

MISCLASSIFIED_PLOT = OUTPUT_DIR / "phase12_misclassified_samples.png"
ERROR_CSV = OUTPUT_DIR / "phase12_error_analysis.csv"

def run_error_analysis():
    print("=" * 60)
    print("          PHASE 12: COMPREHENSIVE ERROR ANALYSIS          ")
    print("=" * 60)

    from phase4_pipeline import create_tf_dataset

    if not FINAL_MODEL_PATH.exists():
        print(f"[ERROR] Model file not found at {FINAL_MODEL_PATH.resolve()}. Run Phase 10 first.")
        sys.exit(1)

    # 1. Load Data & Specs
    with open(CLASS_MAP_JSON, 'r') as f:
        class_map = json.load(f)
        
    inv_class_map = {idx: name for name, idx in class_map.items()}
    test_df = pd.read_csv(TEST_CSV)
    
    model = tf.keras.models.load_model(FINAL_MODEL_PATH)
    test_ds = create_tf_dataset(test_df, class_map, batch_size=32, is_training=False)

    print("\n1. COMPUTING PREDICTIONS & CONFIDENCE SCORES ON TEST SET...")
    y_true = []
    y_pred_probs = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_pred_probs.append(probs)
        y_true.extend(labels.numpy())

    y_pred_probs = np.vstack(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)
    max_confidences = np.max(y_pred_probs, axis=1)
    y_true = np.array(y_true)

    # 2. Build Error Analysis DataFrame
    analysis_records = []
    for idx in range(len(test_df)):
        true_idx = y_true[idx]
        pred_idx = y_pred[idx]
        conf = max_confidences[idx]
        is_correct = (true_idx == pred_idx)
        
        analysis_records.append({
            "filename": test_df.iloc[idx]['filename'],
            "filepath": test_df.iloc[idx]['filepath'],
            "true_class_idx": true_idx,
            "true_class": inv_class_map[true_idx],
            "pred_class_idx": pred_idx,
            "pred_class": inv_class_map[pred_idx],
            "confidence": round(float(conf), 4),
            "is_correct": is_correct
        })

    df_errors = pd.DataFrame(analysis_records)
    df_errors.to_csv(ERROR_CSV, index=False)

    total = len(df_errors)
    correct_cnt = df_errors['is_correct'].sum()
    incorrect_cnt = total - correct_cnt

    print(f"\n2. OVERALL PREDICTION BREAKDOWN:")
    print(f"   Total Test Samples      : {total}")
    print(f"   Correct Predictions     : {correct_cnt} ({correct_cnt/total*100:.2f}%)")
    print(f"   Misclassified Samples   : {incorrect_cnt} ({incorrect_cnt/total*100:.2f}%)")

    # 3. Categorize Errors
    high_conf_errors = df_errors[(~df_errors['is_correct']) & (df_errors['confidence'] >= 0.80)]
    low_conf_predictions = df_errors[df_errors['confidence'] < 0.60]

    print(f"\n3. ERROR CATEGORIZATION:")
    print(f"   High Confidence Errors (Conf >= 80% but WRONG) : {len(high_conf_errors)} samples")
    print(f"   Low Confidence Predictions (Conf < 60%)       : {len(low_conf_predictions)} samples")

    # 4. Most Confused Class Pairs
    print("\n4. TOP MOST FREQUENTLY CONFUSED CLASS PAIRS:")
    misclassified_df = df_errors[~df_errors['is_correct']]
    confused_pairs = misclassified_df.groupby(['true_class', 'pred_class']).size().reset_index(name='count')
    confused_pairs = confused_pairs.sort_values(by='count', ascending=False)
    
    print(confused_pairs.head(8).to_string(index=False))

    # 5. Visualizing Sample Misclassifications
    print("\n5. GENERATING MISCLASSIFIED IMAGES VISUAL GRID...")
    n_plot = min(9, len(misclassified_df))
    if n_plot > 0:
        sample_errors = misclassified_df.sample(n_plot, random_state=42) if len(misclassified_df) >= n_plot else misclassified_df
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 13))
        fig.suptitle("Phase 12: Misclassified Skin Disease Test Samples", fontsize=14, fontweight='bold')
        
        for i, (_, row) in enumerate(sample_errors.iterrows()):
            r, c = i // 3, i % 3
            ax = axes[r, c]
            
            try:
                img = Image.open(row['filepath']).convert('RGB')
                ax.imshow(img)
                title_text = (
                    f"TRUE: {row['true_class']}\n"
                    f"PRED: {row['pred_class']}\n"
                    f"Conf: {row['confidence']*100:.1f}%"
                )
                ax.set_title(title_text, fontsize=9, color='darkred', fontweight='bold')
            except Exception as e:
                ax.text(0.5, 0.5, "Image Read Error", ha='center')
                
            ax.axis('off')
            
        # Hide any unused subplots
        for i in range(n_plot, 9):
            r, c = i // 3, i % 3
            axes[r, c].axis('off')

        plt.tight_layout()
        plt.savefig(MISCLASSIFIED_PLOT, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   Saved misclassified visual grid to: {MISCLASSIFIED_PLOT.resolve()}")

    print("\n" + "=" * 60)
    print("                PHASE 12 ERROR ANALYSIS COMPLETE            ")
    print("=" * 60)
    print(f" Error Analysis CSV Saved: {ERROR_CSV.resolve()}")
    print(" Ready for Phase 13 (Grad-CAM Explainability).")
    print("=" * 60)

if __name__ == "__main__":
    run_error_analysis()
