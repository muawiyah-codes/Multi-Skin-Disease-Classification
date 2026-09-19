import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

OUTPUT_DIR = Path("outputs")
TRAIN_CSV = OUTPUT_DIR / "train_split.csv"
CLASS_MAP_JSON = OUTPUT_DIR / "class_mapping.json"
CLASS_WEIGHTS_JSON = OUTPUT_DIR / "class_weights.json"

def calculate_training_class_weights():
    print("=" * 60)
    print("       PHASE 8: CLASS IMBALANCE WEIGHTS CALCULATION       ")
    print("=" * 60)

    if not TRAIN_CSV.exists() or not CLASS_MAP_JSON.exists():
        print("[ERROR] Required split/mapping files missing. Please run Phases 3 & 4 first.")
        sys.exit(1)

    train_df = pd.read_csv(TRAIN_CSV)
    with open(CLASS_MAP_JSON, 'r') as f:
        class_map = json.load(f)
        
    inv_class_map = {idx: name for name, idx in class_map.items()}
    
    print("\n1. TRAINING SET CLASS COUNTS (STRICTLY TRAINING DATA ONLY):")
    total_train_samples = len(train_df)
    print(f"   Total Training Images: {total_train_samples}")

    train_labels = [class_map[c] for c in train_df['class'].values]
    unique_classes = np.sort(np.unique(train_labels))
    
    # Compute balanced class weights
    # Formula: total_samples / (n_classes * class_samples)
    raw_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=train_labels
    )

    class_weight_dict = {int(cls): float(weight) for cls, weight in zip(unique_classes, raw_weights)}
    
    print("\n2. CALCULATED CLASS WEIGHTS FOR LOSS PENALIZATION:")
    table_data = []
    for cls_idx in unique_classes:
        cname = inv_class_map[cls_idx]
        count = sum(1 for label in train_labels if label == cls_idx)
        pct = (count / total_train_samples) * 100
        w = class_weight_dict[cls_idx]
        table_data.append({
            "Class Index": cls_idx,
            "Class Name": cname,
            "Train Count": count,
            "Percentage (%)": f"{pct:.2f}%",
            "Assigned Weight": f"{w:.4f}"
        })

    df_weights = pd.DataFrame(table_data)
    print(df_weights.to_string(index=False))

    # Save to persistent JSON file
    with open(CLASS_WEIGHTS_JSON, 'w') as f:
        json.dump(class_weight_dict, f, indent=4)
        
    print(f"\n3. SAVED PERSISTENT CLASS WEIGHTS TO: {CLASS_WEIGHTS_JSON.resolve()}")

    print("\n" + "=" * 60)
    print("             PHASE 8 CLASS WEIGHTS VERIFICATION            ")
    print("=" * 60)
    print(" Class weights computed strictly using TRAINING DATA ONLY.")
    print(" Validation and Test data left UN-BALANCED & UN-TOUCHED.")
    print(" Ready for Phase 9 (Feature Extraction Training).")
    print("=" * 60)

if __name__ == "__main__":
    calculate_training_class_weights()
