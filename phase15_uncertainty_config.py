import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

OUTPUT_DIR = Path("outputs")
MODELS_DIR = Path("models")

VAL_CSV = OUTPUT_DIR / "val_split.csv"
CLASS_MAP_JSON = OUTPUT_DIR / "class_mapping.json"
FINAL_MODEL_PATH = MODELS_DIR / "disease_classifier.keras"

UNCERTAINTY_CONFIG_JSON = OUTPUT_DIR / "uncertainty_config.json"
CONF_DIST_PLOT = OUTPUT_DIR / "phase15_confidence_distribution.png"

def calibrate_uncertainty_threshold():
    print("=" * 60)
    print("      PHASE 15: EMPIRICAL UNCERTAINTY THRESHOLD CALIBRATION   ")
    print("=" * 60)

    from phase4_pipeline import create_tf_dataset

    if not FINAL_MODEL_PATH.exists() or not VAL_CSV.exists():
        print("[ERROR] Required model or validation CSV missing. Run Phase 10 first.")
        sys.exit(1)

    # 1. Load Model & Specs
    model = tf.keras.models.load_model(FINAL_MODEL_PATH)
    with open(CLASS_MAP_JSON, 'r') as f:
        class_map = json.load(f)
        
    val_df = pd.read_csv(VAL_CSV)
    print(f"\n1. LOADED VALIDATION DATASET FOR CALIBRATION: {len(val_df)} images")

    val_ds = create_tf_dataset(val_df, class_map, batch_size=32, is_training=False)

    # 2. Run Inference on Validation Dataset
    print("\n2. COMPUTING CONFIDENCE DISTRIBUTIONS ON VALIDATION SET...")
    y_true = []
    y_pred_probs = []

    for images, labels in val_ds:
        probs = model.predict(images, verbose=0)
        y_pred_probs.append(probs)
        y_true.extend(labels.numpy())

    y_pred_probs = np.vstack(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)
    max_confs = np.max(y_pred_probs, axis=1)
    y_true = np.array(y_true)

    is_correct = (y_true == y_pred)

    correct_confs = max_confs[is_correct]
    incorrect_confs = max_confs[~is_correct]

    print("\n3. CONFIDENCE STATS ON VALIDATION DATASET:")
    print(f"   Correct Predictions Mean Conf   : {np.mean(correct_confs)*100:.2f}% (Median: {np.median(correct_confs)*100:.2f}%)")
    print(f"   Incorrect Predictions Mean Conf : {np.mean(incorrect_confs)*100:.2f}% (Median: {np.median(incorrect_confs)*100:.2f}%)")

    # 4. Calibrate Threshold
    # We want a threshold tau where predictions >= tau have ultra-high reliability (>98% accuracy)
    possible_thresholds = np.arange(0.50, 0.95, 0.05)
    calib_table = []
    
    selected_threshold = 0.70 # Default fallback
    
    for tau in possible_thresholds:
        mask = max_confs >= tau
        accepted_cnt = np.sum(mask)
        if accepted_cnt > 0:
            acc_above_tau = np.mean(is_correct[mask])
            rejection_pct = (1.0 - (accepted_cnt / len(val_df))) * 100
            calib_table.append({
                "Threshold (Tau)": round(float(tau), 2),
                "Accepted Images": accepted_cnt,
                "Rejection Rate (%)": f"{rejection_pct:.1f}%",
                "Accuracy Above Tau (%)": f"{acc_above_tau*100:.2f}%"
            })
            if acc_above_tau >= 0.975 and selected_threshold == 0.70:
                selected_threshold = round(float(tau), 2)

    df_calib = pd.DataFrame(calib_table)
    print("\n4. THRESHOLD CALIBRATION SWEEP TABLE:")
    print(df_calib.to_string(index=False))

    print(f"\n   --> SELECTED EMPIRICAL UNCERTAINTY THRESHOLD (Tau): {selected_threshold}")

    # 5. Plot Confidence Distributions
    print("\n5. PLOTTING CONFIDENCE DISTRIBUTION DENSITY...")
    plt.figure(figsize=(10, 5))
    sns.kdeplot(correct_confs, color='green', label='Correct Predictions', fill=True, alpha=0.3)
    sns.kdeplot(incorrect_confs, color='red', label='Incorrect Predictions', fill=True, alpha=0.3)
    plt.axvline(selected_threshold, color='darkblue', linestyle='--', linewidth=2, label=f'Calibrated Tau = {selected_threshold}')
    plt.title("Phase 15: Validation Set Softmax Confidence Distribution", fontsize=13, fontweight='bold')
    plt.xlabel("Max Softmax Confidence Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CONF_DIST_PLOT, dpi=150)
    plt.close()
    print(f"   Saved Confidence Distribution Plot to: {CONF_DIST_PLOT.resolve()}")

    # 6. Save Persistent Calibration JSON Config
    config_data = {
        "validator_threshold": 0.50,
        "uncertainty_threshold": selected_threshold,
        "calibration_dataset": "val_split.csv",
        "calibrated_accuracy_above_threshold": float(df_calib[df_calib['Threshold (Tau)']==selected_threshold]['Accuracy Above Tau (%)'].values[0].replace('%','')) / 100.0,
        "rejection_message_invalid": "Please upload a valid skin lesion image. This image is outside the supported input domain.",
        "rejection_message_uncertain": "The model cannot make a sufficiently reliable classification for this image."
    }

    with open(UNCERTAINTY_CONFIG_JSON, 'w') as f:
        json.dump(config_data, f, indent=4)

    print(f"\n6. SAVED UNCERTAINTY CONFIG TO: {UNCERTAINTY_CONFIG_JSON.resolve()}")

    print("\n" + "=" * 60)
    print("          PHASE 15 UNCERTAINTY CALIBRATION COMPLETE         ")
    print("=" * 60)
    print(f" Calibrated Threshold Tau: {selected_threshold}")
    print(" Dual-Gate Protection Ready (Gate 1: Validator, Gate 2: Confidence Threshold).")
    print(" Ready for Phase 16 (Single Image Inference Module).")
    print("=" * 60)

if __name__ == "__main__":
    calibrate_uncertainty_threshold()
