import os
import sys
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

np.random.seed(42)

DATASET_DIR = Path(r"d:\Skin Disease classification\newtrain")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def get_file_hash(filepath, chunk_size=65536):
    """Calculate SHA256 hash of a file to detect duplicates."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()

def audit_dataset(dataset_dir):
    print("=" * 60)
    print("         PHASE 1: DATASET QUALITY AUDIT LOG         ")
    print("=" * 60)
    
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset path does not exist: {dataset_dir}")
        sys.exit(1)
        
    class_folders = [f for f in dataset_dir.iterdir() if f.is_dir()]
    class_names = sorted([f.name for f in class_folders])
    num_classes = len(class_names)
    
    print(f"\n1. DIRECTORY STRUCTURE & CLASSES:")
    print(f"   Dataset Root: {dataset_dir.resolve()}")
    print(f"   Total Classes Found: {num_classes}")
    print(f"   Class Names:")
    for idx, cname in enumerate(class_names, 1):
        print(f"     {idx}. {cname}")
        
    if num_classes == 0:
        print("[ERROR] No subdirectories (classes) found in dataset directory!")
        sys.exit(1)

    # Gather all file paths first
    all_files_to_process = []
    for class_folder in class_folders:
        cname = class_folder.name
        files = [f for f in class_folder.glob("**/*") if f.is_file()]
        for f in files:
            all_files_to_process.append((cname, f))
            
    total_files = len(all_files_to_process)
    print(f"\n2. AUDITING {total_files} IMAGES (Integrity, Dimensions, Hashes)...")
    sys.stdout.flush()

    class_counts = Counter()
    file_extensions = Counter()
    corrupted_files = []
    image_sizes = []
    channels_count = Counter()
    low_res_files = []
    hashes = defaultdict(list)
    class_image_map = defaultdict(list)
    
    for idx, (cname, fpath) in enumerate(all_files_to_process, 1):
        if idx % 500 == 0 or idx == total_files:
            print(f"   Processed {idx}/{total_files} images...")
            sys.stdout.flush()
            
        ext = fpath.suffix.lower()
        file_extensions[ext] += 1
        
        if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']:
            continue
            
        class_counts[cname] += 1
        class_image_map[cname].append(fpath)
        
        # Hash check
        try:
            fhash = get_file_hash(fpath)
            hashes[fhash].append(fpath)
        except Exception as e:
            corrupted_files.append((str(fpath), f"Hash read error: {str(e)}"))
            continue

        # PIL Image Verification & stats
        try:
            with Image.open(fpath) as img:
                img.verify()
            with Image.open(fpath) as img:
                w, h = img.size
                mode = img.mode
                image_sizes.append((w, h))
                channels_count[mode] += 1
                if w < 224 or h < 224:
                    low_res_files.append((str(fpath), w, h))
        except Exception as e:
            corrupted_files.append((str(fpath), f"Corrupted image: {str(e)}"))

    total_images = sum(class_counts.values())
    
    print(f"\n3. CLASS DISTRIBUTION STATS:")
    print(f"   Total Valid Images Analyzed: {total_images}")
    
    if total_images == 0:
        print("[ERROR] No valid images found!")
        sys.exit(1)
        
    df_counts = pd.DataFrame([
        {"Class": cname, "Count": class_counts[cname], "Percentage (%)": round((class_counts[cname]/total_images)*100, 2)}
        for cname in class_names
    ])
    print(df_counts.to_string(index=False))
    
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    print(f"\n   Imbalance Ratio (Max / Min): {imbalance_ratio:.2f}")

    print(f"\n4. FILE EXTENSIONS DISCOVERY:")
    for ext, cnt in file_extensions.items():
        print(f"   {ext}: {cnt} files")

    print(f"\n5. CORRUPTED IMAGES AUDIT:")
    print(f"   Total Corrupted Images Found: {len(corrupted_files)}")
    if corrupted_files:
        for cfile, err in corrupted_files[:10]:
            print(f"     - {cfile}: {err}")
        if len(corrupted_files) > 10:
            print(f"     ... and {len(corrupted_files)-10} more.")

    print(f"\n6. IMAGE RESOLUTION & CHANNEL METRICS:")
    if image_sizes:
        widths = [s[0] for s in image_sizes]
        heights = [s[1] for s in image_sizes]
        print(f"   Width  - Min: {min(widths)}, Max: {max(widths)}, Mean: {np.mean(widths):.1f}, Median: {np.median(widths):.1f}")
        print(f"   Height - Min: {min(heights)}, Max: {max(heights)}, Mean: {np.mean(heights):.1f}, Median: {np.median(heights):.1f}")
        print(f"   Channels Distribution: {dict(channels_count)}")
        print(f"   Low-resolution (<224x224): {len(low_res_files)} images ({len(low_res_files)/total_images*100:.2f}%)")

    print(f"\n7. DUPLICATE IMAGES AUDIT (SHA256 Hash Matching):")
    duplicate_groups = {h: paths for h, paths in hashes.items() if len(paths) > 1}
    total_duplicate_files = sum(len(paths) - 1 for paths in duplicate_groups.values())
    print(f"   Unique Image Hashes: {len(hashes)}")
    print(f"   Duplicate Groups Found: {len(duplicate_groups)}")
    print(f"   Total Redundant/Duplicate Files: {total_duplicate_files}")
    
    if duplicate_groups:
        print("   Sample Duplicate Sets:")
        for idx, (h, paths) in enumerate(list(duplicate_groups.items())[:5], 1):
            print(f"     Set {idx}: {len(paths)} copies")
            for p in paths[:3]:
                print(f"       - {p}")

    # Generate Random Samples Visualization
    print(f"\n8. GENERATING SAMPLE VISUALIZATION GRID FOR MANUAL INSPECTION...")
    samples_per_class = 3
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(3 * samples_per_class, 3 * num_classes))
    fig.suptitle("Phase 1: Random Dataset Image Samples per Class", fontsize=14, fontweight='bold')
    
    for row, cname in enumerate(class_names):
        imgs = class_image_map[cname]
        selected_imgs = np.random.choice(imgs, size=min(samples_per_class, len(imgs)), replace=False)
        
        for col in range(samples_per_class):
            ax = axes[row, col] if num_classes > 1 else axes[col]
            if col < len(selected_imgs):
                img_path = selected_imgs[col]
                try:
                    img = Image.open(img_path).convert('RGB')
                    ax.imshow(img)
                    ax.set_title(f"{cname}\n({img.width}x{img.height})", fontsize=8)
                except Exception as e:
                    ax.text(0.5, 0.5, 'Error Loading', ha='center', va='center')
            ax.axis('off')
            
    plt.tight_layout()
    sample_out_path = OUTPUT_DIR / "phase1_class_samples.png"
    plt.savefig(sample_out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved random sample grid visualization to: {sample_out_path.resolve()}")

    # Determine Verdict
    print("\n" + "=" * 60)
    print("                    DATASET QUALITY REPORT                  ")
    print("=" * 60)
    print(f" Total Classes: {num_classes}")
    print(f" Total Images: {total_images}")
    print(f" Corrupted Images: {len(corrupted_files)}")
    print(f" Exact Duplicates: {total_duplicate_files}")
    print(f" Low-Res Images (<224x224): {len(low_res_files)}")
    print(f" Imbalance Ratio: {imbalance_ratio:.2f}")
    
    warnings = []
    if len(corrupted_files) > 0:
        warnings.append(f"{len(corrupted_files)} corrupted files detected.")
    if total_duplicate_files > 0:
        warnings.append(f"{total_duplicate_files} exact duplicate image files detected.")
    if len(low_res_files) > 0:
        warnings.append(f"{len(low_res_files)} images have dimensions smaller than 224x224.")
    if imbalance_ratio > 1.5:
        warnings.append(f"Dataset has noticeable class imbalance (ratio {imbalance_ratio:.2f}).")
        
    print("-" * 60)
    if len(corrupted_files) > 0 or total_images < 100:
        status = "FAIL"
    elif len(warnings) > 0:
        status = "PASS WITH WARNINGS"
    else:
        status = "PASS"
        
    print(f" FINAL STATUS: [{status}]")
    print("-" * 60)
    if warnings:
        print(" Summary of Warnings/Issues:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print(" Clean dataset! Ready for Phase 2.")
    print("=" * 60)
    sys.stdout.flush()

if __name__ == "__main__":
    audit_dataset(DATASET_DIR)
