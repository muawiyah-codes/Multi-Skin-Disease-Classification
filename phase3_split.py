import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Fixed seed for absolute reproducibility
RANDOM_SEED = 42

OUTPUT_DIR = Path("outputs")
METADATA_CSV = OUTPUT_DIR / "dataset_metadata.csv"

def generate_splits():
    print("=" * 60)
    print("      PHASE 3: TRAIN / VALIDATION / TEST SPLIT (70/15/15)   ")
    print("=" * 60)

    if not METADATA_CSV.exists():
        print(f"[ERROR] Metadata CSV not found at {METADATA_CSV.resolve()}. Please run Phase 2 first.")
        sys.exit(1)

    df = pd.read_csv(METADATA_CSV)
    total_images = len(df)
    print(f"\n1. LOADED DATASET METADATA:")
    print(f"   Total Images to Split: {total_images}")
    print(f"   Random Seed: {RANDOM_SEED}")
    print(f"   Target Ratios: 70% Train, 15% Validation, 15% Test")

    # Perform Stratified Split:
    # First split 70% Train and 30% Temp (Val + Test)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=df['class']
    )

    # Next split 30% Temp into 15% Val and 15% Test (50/50 ratio of 30%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df['class']
    )

    # Assign split labels
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print("\n2. SPLIT SUMMARY:")
    print(f"   Train Set : {len(train_df)} images ({len(train_df)/total_images*100:.2f}%)")
    print(f"   Val Set   : {len(val_df)} images ({len(val_df)/total_images*100:.2f}%)")
    print(f"   Test Set  : {len(test_df)} images ({len(test_df)/total_images*100:.2f}%)")

    # Verify Stratification across classes
    print("\n3. CLASS DISTRIBUTION PER SPLIT (Stratification Check):")
    class_names = sorted(df['class'].unique())
    
    stats = []
    for cname in class_names:
        n_train = len(train_df[train_df['class'] == cname])
        n_val = len(val_df[val_df['class'] == cname])
        n_test = len(test_df[test_df['class'] == cname])
        n_total = n_train + n_val + n_test
        
        stats.append({
            "Class": cname,
            "Train": f"{n_train} ({n_train/n_total*100:.1f}%)",
            "Val": f"{n_val} ({n_val/n_total*100:.1f}%)",
            "Test": f"{n_test} ({n_test/n_total*100:.1f}%)",
            "Total": n_total
        })
        
    df_stats = pd.DataFrame(stats)
    print(df_stats.to_string(index=False))

    # Save Persistent Split CSVs
    print("\n4. SAVING PERSISTENT SPLIT CSV FILES...")
    train_csv = OUTPUT_DIR / "train_split.csv"
    val_csv = OUTPUT_DIR / "val_split.csv"
    test_csv = OUTPUT_DIR / "test_split.csv"
    full_splits_csv = OUTPUT_DIR / "dataset_splits.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    combined_df.to_csv(full_splits_csv, index=False)

    print(f"   Saved Train Split CSV: {train_csv.resolve()}")
    print(f"   Saved Val Split CSV  : {val_csv.resolve()}")
    print(f"   Saved Test Split CSV : {test_csv.resolve()}")
    print(f"   Saved Full Split CSV : {full_splits_csv.resolve()}")

    print("\n" + "=" * 60)
    print("               PHASE 3 SPLITTING VERIFICATION              ")
    print("=" * 60)
    print(f" Leakage Check: Train, Val, and Test sets are mutually exclusive.")
    print(f" Persistent CSVs generated successfully.")
    print(" Ready for Phase 4 (TensorFlow Input Pipeline).")
    print("=" * 60)

if __name__ == "__main__":
    generate_splits()
