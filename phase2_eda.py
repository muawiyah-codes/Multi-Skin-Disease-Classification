import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for professional figures
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 11})

DATASET_DIR = Path(r"d:\Skin Disease classification\newtrain")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def run_eda(dataset_dir):
    print("=" * 60)
    print("         PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)        ")
    print("=" * 60)

    class_folders = [f for f in dataset_dir.iterdir() if f.is_dir()]
    class_names = sorted([f.name for f in class_folders])
    
    records = []
    
    print("\n1. COLLECTING METADATA FROM ALL IMAGES...")
    for class_folder in class_folders:
        cname = class_folder.name
        files = [f for f in class_folder.glob("**/*") if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        for fpath in files:
            try:
                with Image.open(fpath) as img:
                    w, h = img.size
                    mode = img.mode
                    aspect_ratio = round(w / h, 3)
                    records.append({
                        "filename": fpath.name,
                        "class": cname,
                        "filepath": str(fpath),
                        "width": w,
                        "height": h,
                        "aspect_ratio": aspect_ratio,
                        "channels": mode
                    })
            except Exception as e:
                continue

    df = pd.DataFrame(records)
    print(f"   Successfully parsed metadata for {len(df)} images.")
    
    # Save CSV metadata for Phase 3 and future use
    csv_path = OUTPUT_DIR / "dataset_metadata.csv"
    df.to_csv(csv_path, index=False)
    print(f"   Saved metadata CSV to: {csv_path.resolve()}")

    # 1. Class Distribution Analysis
    print("\n2. GENERATING CLASS DISTRIBUTION PLOT...")
    plt.figure(figsize=(12, 6))
    class_counts = df['class'].value_counts().reindex(class_names)
    total_imgs = len(df)
    
    ax = sns.barplot(x=class_counts.values, y=class_counts.index, palette="viridis")
    plt.title("Class Distribution (Total: 9,888 Images)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Number of Images", fontsize=12)
    plt.ylabel("Skin Disease Class", fontsize=12)
    
    # Add count & percentage annotations on bars
    for i, count in enumerate(class_counts.values):
        pct = (count / total_imgs) * 100
        ax.text(count + 15, i, f"{count} ({pct:.1f}%)", va='center', fontsize=10, fontweight='bold')
        
    plt.xlim(0, max(class_counts.values) * 1.18)
    plt.tight_layout()
    chart1_path = OUTPUT_DIR / "eda_class_distribution.png"
    plt.savefig(chart1_path, dpi=150)
    plt.close()
    print(f"   Saved Class Distribution Plot to: {chart1_path.resolve()}")

    # 2. Image Dimensions Distribution
    print("\n3. GENERATING IMAGE DIMENSIONS DISTRIBUTION PLOT...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot Width vs Height
    sns.scatterplot(data=df, x='width', y='height', hue='class', alpha=0.5, ax=axes[0], s=25, legend=False)
    axes[0].axvline(224, color='red', linestyle='--', linewidth=1.5, label='Target Size 224x224')
    axes[0].axhline(224, color='red', linestyle='--', linewidth=1.5)
    axes[0].set_title("Width vs Height Scatter Plot", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Width (pixels)")
    axes[0].set_ylabel("Height (pixels)")
    axes[0].legend()
    
    # Histogram of Widths & Heights
    sns.kdeplot(df['width'], ax=axes[1], color='blue', label='Width', fill=True, alpha=0.3)
    sns.kdeplot(df['height'], ax=axes[1], color='orange', label='Height', fill=True, alpha=0.3)
    axes[1].set_title("Width & Height Density Distribution", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Pixels")
    axes[1].legend()
    
    plt.tight_layout()
    chart2_path = OUTPUT_DIR / "eda_image_dimensions.png"
    plt.savefig(chart2_path, dpi=150)
    plt.close()
    print(f"   Saved Image Dimensions Plot to: {chart2_path.resolve()}")

    # 3. Aspect Ratio Distribution
    print("\n4. GENERATING ASPECT RATIO DISTRIBUTION PLOT...")
    plt.figure(figsize=(10, 5))
    sns.histplot(df['aspect_ratio'], bins=40, kde=True, color='teal')
    plt.axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='Square (1:1 Aspect Ratio)')
    plt.axvline(1.33, color='orange', linestyle='--', linewidth=1.5, label='4:3 Aspect Ratio')
    plt.title("Aspect Ratio (Width / Height) Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Aspect Ratio (W / H)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    chart3_path = OUTPUT_DIR / "eda_aspect_ratios.png"
    plt.savefig(chart3_path, dpi=150)
    plt.close()
    print(f"   Saved Aspect Ratio Plot to: {chart3_path.resolve()}")

    # 4. Summary Table Output
    print("\n" + "=" * 60)
    print("                     EDA SUMMARY STATISTICS                 ")
    print("=" * 60)
    print(f" Total Images Evaluated: {len(df)}")
    print(f" Total Unique Classes: {len(class_names)}")
    print(f" Width Stats  -> Min: {df['width'].min()}, Max: {df['width'].max()}, Mean: {df['width'].mean():.1f}, Median: {df['width'].median():.1f}")
    print(f" Height Stats -> Min: {df['height'].min()}, Max: {df['height'].max()}, Mean: {df['height'].mean():.1f}, Median: {df['height'].median():.1f}")
    print(f" Aspect Ratio -> Min: {df['aspect_ratio'].min():.2f}, Max: {df['aspect_ratio'].max():.2f}, Median: {df['aspect_ratio'].median():.2f}")
    print(f" Square Images (AR = 1.0 +/- 0.05): {len(df[np.isclose(df['aspect_ratio'], 1.0, atol=0.05)])} ({len(df[np.isclose(df['aspect_ratio'], 1.0, atol=0.05)])/len(df)*100:.1f}%)")
    print("=" * 60)

if __name__ == "__main__":
    run_eda(DATASET_DIR)
