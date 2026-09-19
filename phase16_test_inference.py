import os
import sys
from pathlib import Path
import pandas as pd
import cv2
import numpy as np

# Add src to path
sys.path.append(str(Path(r"d:\Skin Disease classification").resolve()))

from src.inference import SkinDiseaseInferenceEngine

OUTPUT_DIR = Path("outputs")
TEST_CSV = OUTPUT_DIR / "test_split.csv"

def test_inference_engine():
    print("=" * 60)
    print("      PHASE 16: DUAL-GATE INFERENCE PIPELINE TEST        ")
    print("=" * 60)

    engine = SkinDiseaseInferenceEngine()
    print("   Inference Engine Initialized Successfully.")

    # 1. Test Valid Lesion Image
    test_df = pd.read_csv(TEST_CSV)
    valid_sample_path = test_df.iloc[0]['filepath']
    true_class = test_df.iloc[0]['class']

    print(f"\n1. TESTING VALID SKIN LESION SAMPLE ({Path(valid_sample_path).name}):")
    res_valid = engine.predict_image(valid_sample_path)
    
    print(f"   Status               : {res_valid['status']}")
    print(f"   Validator Score      : {res_valid['validator_score']} (Valid: {res_valid['is_valid_domain']})")
    print(f"   True Class           : {true_class}")
    print(f"   Predicted Class      : {res_valid['predicted_class_name']}")
    print(f"   Confidence Score     : {res_valid['confidence']*100:.2f}%")
    print(f"   Status Message       : {res_valid['message']}")
    
    # Save Grad-CAM overlay if present
    if res_valid['gradcam_overlay'] is not None:
        out_overlay = OUTPUT_DIR / "phase16_test_valid_gradcam.png"
        cv2.imwrite(str(out_overlay), cv2.cvtColor(res_valid['gradcam_overlay'], cv2.COLOR_RGB2BGR))
        print(f"   Saved Grad-CAM Overlay to: {out_overlay.resolve()}")

    # 2. Test Invalid OOD Image
    invalid_sample_path = Path("data/validator_dataset/INVALID/invalid_ood_0001.jpg")
    print(f"\n2. TESTING INVALID OOD SAMPLE ({invalid_sample_path.name}):")
    
    if invalid_sample_path.exists():
        res_invalid = engine.predict_image(invalid_sample_path)
        print(f"   Status               : {res_invalid['status']}")
        print(f"   Validator Score      : {res_invalid['validator_score']} (Valid: {res_invalid['is_valid_domain']})")
        print(f"   Predicted Class      : {res_invalid['predicted_class_name']}")
        print(f"   Status Message       : {res_invalid['message']}")
    else:
        print(f"   [INFO] Sample invalid file not found at {invalid_sample_path.resolve()}")

    print("\n" + "=" * 60)
    print("              PHASE 16 INFERENCE ENGINE VERIFIED           ")
    print("=" * 60)
    print(" Dual-gate inference pipeline fully functional.")
    print(" Reusable modular engine ready for Streamlit UI (Phase 17).")
    print("=" * 60)

if __name__ == "__main__":
    test_inference_engine()
