import os
import sys
import json
from pathlib import Path
import tensorflow as tf
import numpy as np

OUTPUT_DIR = Path("outputs")
MODELS_DIR = Path("models")

DISEASE_MODEL_PATH = MODELS_DIR / "disease_classifier.keras"
VALIDATOR_MODEL_PATH = MODELS_DIR / "image_validator.keras"
CLASS_MAP_JSON = OUTPUT_DIR / "class_mapping.json"
UNCERTAINTY_CONFIG_JSON = OUTPUT_DIR / "uncertainty_config.json"
CLASS_WEIGHTS_JSON = OUTPUT_DIR / "class_weights.json"

def verify_all_models_and_artifacts():
    print("=" * 60)
    print("      PHASE 18: SAVE MODELS & ARTIFACTS VERIFICATION      ")
    print("=" * 60)

    # 1. Verify File Existence & Sizes
    artifacts = [
        ("Disease Classifier Model", DISEASE_MODEL_PATH),
        ("Image Validator Model", VALIDATOR_MODEL_PATH),
        ("Class Mapping JSON", CLASS_MAP_JSON),
        ("Uncertainty Config JSON", UNCERTAINTY_CONFIG_JSON),
        ("Class Weights JSON", CLASS_WEIGHTS_JSON),
    ]

    print("\n1. AUDITING ARTIFACT FILES ON DISK:")
    all_exist = True
    for name, path in artifacts:
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"   [OK] {name:<26} -> {path.name:<25} ({size_mb:.2f} MB)")
        else:
            print(f"   [MISSING] {name:<26} -> {path.name:<25}")
            all_exist = False

    if not all_exist:
        print("\n[ERROR] Some required artifacts are missing!")
        sys.exit(1)

    # 2. Test Reloading Disease Classifier Model
    print(f"\n2. RELOADING DISEASE CLASSIFIER MODEL ({DISEASE_MODEL_PATH.name})...")
    disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)
    print("   [SUCCESS] Disease Classifier loaded successfully.")
    
    # Test dummy prediction
    dummy_input = tf.random.uniform((1, 224, 224, 3), minval=0, maxval=255, dtype=tf.float32)
    preds = disease_model.predict(dummy_input, verbose=0)
    print(f"   Reload Prediction Output Shape : {preds.shape} (Expected: (1, 9))")
    print(f"   Probabilities Sum              : {np.sum(preds):.4f} (Expected: 1.0)")
    assert preds.shape == (1, 9), "Mismatch in output shape!"
    assert np.isclose(np.sum(preds), 1.0, atol=1e-3), "Probabilities do not sum to 1.0!"

    # 3. Test Reloading Image Validator Model
    print(f"\n3. RELOADING IMAGE VALIDATOR MODEL ({VALIDATOR_MODEL_PATH.name})...")
    validator_model = tf.keras.models.load_model(VALIDATOR_MODEL_PATH)
    print("   [SUCCESS] Image Validator loaded successfully.")
    
    val_preds = validator_model.predict(dummy_input, verbose=0)
    print(f"   Reload Validator Output Shape  : {val_preds.shape} (Expected: (1, 1))")
    print(f"   Validator Score Output         : {val_preds[0][0]:.4f}")
    assert val_preds.shape == (1, 1), "Mismatch in validator output shape!"

    # 4. Verify Configuration Content
    print("\n4. VERIFYING CONFIGURATION INTEGRITY:")
    with open(CLASS_MAP_JSON, 'r') as f:
        cmap = json.load(f)
    with open(UNCERTAINTY_CONFIG_JSON, 'r') as f:
        uconfig = json.load(f)

    print(f"   Total Verified Classes     : {len(cmap)}")
    print(f"   Uncertainty Threshold Tau  : {uconfig.get('uncertainty_threshold')}")
    print(f"   Validator Threshold        : {uconfig.get('validator_threshold')}")

    print("\n" + "=" * 60)
    print("             PHASE 18 ARTIFACTS VERIFICATION COMPLETE       ")
    print("=" * 60)
    print(" All models saved in modern Keras v3 format (.keras).")
    print(" Reload prediction tests passed with 100% integrity.")
    print(" Ready for Phase 19 (Final Project Directory Organization).")
    print("=" * 60)

if __name__ == "__main__":
    verify_all_models_and_artifacts()
