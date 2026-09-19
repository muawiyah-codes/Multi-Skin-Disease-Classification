import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

OUTPUT_DIR = Path("outputs")
MODELS_DIR = Path("models")

TEST_CSV = OUTPUT_DIR / "test_split.csv"
CLASS_MAP_JSON = OUTPUT_DIR / "class_mapping.json"
FINAL_MODEL_PATH = MODELS_DIR / "disease_classifier.keras"
GRADCAM_PLOT = OUTPUT_DIR / "phase13_gradcam_overlay.png"

def make_gradcam_heatmap(img_array, full_model, last_conv_layer_name="top_conv", pred_index=None):
    """
    Computes Grad-CAM heatmap for EfficientNetV2B0 model.
    Handles nested functional base model structure cleanly.
    """
    # Find base model inside full model
    base_model = None
    for layer in full_model.layers:
        if "efficientnet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("EfficientNet base layer not found in model!")

    # Find last convolutional layer inside base model
    target_conv_layer = None
    for layer in reversed(base_model.layers):
        if layer.name == last_conv_layer_name or isinstance(layer, tf.keras.layers.Conv2D):
            target_conv_layer = layer
            break

    if target_conv_layer is None:
        raise ValueError(f"Conv layer {last_conv_layer_name} not found in base model!")

    # Build sub-model that maps base model input -> (last conv output, final output)
    grad_model = tf.keras.models.Model(
        inputs=[base_model.inputs],
        outputs=[target_conv_layer.output, base_model.output]
    )

    # Record operations for GradientTape
    with tf.GradientTape() as tape:
        # Pass through data augmentation layer if present, or feed directly to base model
        conv_outputs, base_outputs = grad_model(img_array)
        
        # Pass base_outputs through classification head layers
        x = base_outputs
        for layer in full_model.layers:
            if layer.name in ["global_avg_pooling", "head_batch_norm", "head_dense_256", "head_dropout", "output_softmax"]:
                x = layer(x)
                
        preds = x
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Compute gradients of class score w.r.t. feature map outputs
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Global average pooling of gradients = neuron importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply feature maps by importance weights
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU (keep only positive activations influencing predicted class)
    heatmap = tf.maximum(heatmap, 0.0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def generate_gradcam_visualizations():
    print("=" * 60)
    print("       PHASE 13: GRAD-CAM VISUAL EXPLAINABILITY PIPELINE     ")
    print("=" * 60)

    from phase4_pipeline import parse_image

    if not FINAL_MODEL_PATH.exists():
        print(f"[ERROR] Final model not found at {FINAL_MODEL_PATH.resolve()}. Run Phase 10 first.")
        sys.exit(1)

    # 1. Load Specs & Model
    with open(CLASS_MAP_JSON, 'r') as f:
        class_map = json.load(f)
        
    inv_class_map = {idx: name for name, idx in class_map.items()}
    class_names = [inv_class_map[i] for i in range(len(class_map))]
    
    model = tf.keras.models.load_model(FINAL_MODEL_PATH)
    test_df = pd.read_csv(TEST_CSV)

    # Pick 4 diverse sample images from test set (including Melanoma, Benign, Eczema)
    sample_indices = []
    for cname in ['Melanoma', 'Benign keratosis', 'Atopic Dermatitis', 'Vascular lesion']:
        sub = test_df[test_df['class'] == cname]
        if len(sub) > 0:
            sample_indices.append(sub.index[0])

    print(f"\n1. GENERATING GRAD-CAM FOR {len(sample_indices)} DIVERSE TEST SAMPLES...")

    fig, axes = plt.subplots(len(sample_indices), 3, figsize=(12, 4 * len(sample_indices)))
    fig.suptitle("Phase 13: Grad-CAM Feature Attribution & Attention Overlay", fontsize=14, fontweight='bold')

    for row_idx, idx in enumerate(sample_indices):
        img_path = test_df.iloc[idx]['filepath']
        true_class = test_df.iloc[idx]['class']
        
        # Load raw tensor & float image
        img_tensor, _ = parse_image(img_path, 0)
        img_batch = tf.expand_dims(img_tensor, axis=0) # (1, 224, 224, 3)

        # Predict
        preds = model.predict(img_batch, verbose=0)
        pred_idx = np.argmax(preds[0])
        pred_class = inv_class_map[pred_idx]
        conf = preds[0][pred_idx]

        # Generate Grad-CAM Heatmap
        heatmap = make_gradcam_heatmap(img_batch, model, last_conv_layer_name="top_conv", pred_index=pred_idx)
        
        # Original Image uint8
        orig_img = img_tensor.numpy().astype("uint8")
        
        # Colorize Heatmap with JET colormap
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
        
        # Create Overlay (0.6 original + 0.4 heatmap)
        overlay = cv2.addWeighted(orig_img, 0.6, color_heatmap, 0.4, 0)

        # Plot Column 1: Original
        ax1 = axes[row_idx, 0] if len(sample_indices) > 1 else axes[0]
        ax1.imshow(orig_img)
        ax1.set_title(f"Original Image\nTrue: {true_class}", fontsize=10)
        ax1.axis('off')

        # Plot Column 2: Heatmap
        ax2 = axes[row_idx, 1] if len(sample_indices) > 1 else axes[1]
        ax2.imshow(color_heatmap)
        ax2.set_title(f"Grad-CAM Heatmap\n(Layer: top_conv)", fontsize=10)
        ax2.axis('off')

        # Plot Column 3: Overlay
        ax3 = axes[row_idx, 2] if len(sample_indices) > 1 else axes[2]
        ax3.imshow(overlay)
        ax3.set_title(f"Attention Overlay\nPred: {pred_class} ({conf*100:.1f}%)", fontsize=10, color='darkgreen', fontweight='bold')
        ax3.axis('off')

    plt.tight_layout()
    plt.savefig(GRADCAM_PLOT, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   Saved Grad-CAM Overlay Panel to: {GRADCAM_PLOT.resolve()}")

    print("\n" + "=" * 60)
    print("                MEDICAL EXPLAINABILITY DISCLAIMER            ")
    print("=" * 60)
    print(" IMPORTANT: Grad-CAM shows neural network gradient activation focus.")
    print(" It DOES NOT prove medical diagnostic correctness, clinical validity,")
    print(" or histological proof. It is a visual attribution tool only.")
    print(" Ready for Phase 14 (Invalid Image / OOD Protection).")
    print("=" * 60)

if __name__ == "__main__":
    generate_gradcam_visualizations()
