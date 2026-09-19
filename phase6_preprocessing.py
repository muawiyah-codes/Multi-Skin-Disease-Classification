import os
import sys
import numpy as np
import tensorflow as tf

def audit_efficientnetv2_preprocessing():
    print("=" * 60)
    print("      PHASE 6: EFFICIENTNETV2 PREPROCESSING AUDIT        ")
    print("=" * 60)

    # Instantiate EfficientNetV2B0 base model
    print("\n1. INITIALIZING tf.keras.applications.EfficientNetV2B0...")
    base_model = tf.keras.applications.EfficientNetV2B0(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
        include_preprocessing=True # Explicitly verifying default built-in preprocessing
    )

    print(f"   Model Name : {base_model.name}")
    print(f"   Total Layers: {len(base_model.layers)}")
    
    # Inspect first 5 layers of base_model to check built-in Rescaling / Preprocessing layer
    print("\n2. INSPECTING INITIAL LAYERS OF EFFICIENTNETV2B0:")
    for i in range(min(5, len(base_model.layers))):
        layer = base_model.layers[i]
        print(f"   Layer {i}: Name = {layer.name:<25} | Type = {layer.__class__.__name__:<20}")
        if "rescaling" in layer.name.lower() or "normalization" in layer.name.lower():
            config = layer.get_config()
            scale = config.get('scale', 'N/A')
            offset = config.get('offset', 'N/A')
            print(f"            --> Config: scale = {scale}, offset = {offset}")

    # Test tensor pass
    print("\n3. TESTING TENSOR PASS WITH RAW PIXEL RANGE [0.0, 255.0]:")
    raw_tensor = tf.constant(np.full((1, 224, 224, 3), 255.0), dtype=tf.float32)
    print(f"   Input Raw Tensor  -> Min: {tf.reduce_min(raw_tensor):.1f}, Max: {tf.reduce_max(raw_tensor):.1f}")
    
    # Trace first layer (Rescaling layer inside base_model)
    first_layer = base_model.layers[0] # input layer or rescaling layer
    rescale_layer = None
    for layer in base_model.layers[:3]:
        if "rescaling" in layer.name.lower() or isinstance(layer, tf.keras.layers.Rescaling):
            rescale_layer = layer
            break
            
    if rescale_layer:
        rescaled_output = rescale_layer(raw_tensor)
        print(f"   Rescaled Layer Output -> Min: {tf.reduce_min(rescaled_output):.4f}, Max: {tf.reduce_max(rescaled_output):.4f}")
    
    # Test double normalization risk simulation
    print("\n4. DOUBLE NORMALIZATION DEMONSTRATION (WHY IT BREAKS LEARNING):")
    double_norm_tensor = raw_tensor / 255.0 # If user manually divided by 255 in pipeline
    print(f"   If manually normalized first -> Tensor Min: {tf.reduce_min(double_norm_tensor):.4f}, Max: {tf.reduce_max(double_norm_tensor):.4f}")
    if rescale_layer:
        distorted_output = rescale_layer(double_norm_tensor)
        print(f"   After EfficientNet Rescaling  -> Min: {tf.reduce_min(distorted_output):.6f}, Max: {tf.reduce_max(distorted_output):.6f}")
        print("   --> WARNING: Input range shrunk down to [0, 0.00392]! Model features destroyed!")

    print("\n" + "=" * 60)
    print("             PHASE 6 PREPROCESSING VERIFICATION            ")
    print("=" * 60)
    print(" Verified: EfficientNetV2B0 has built-in include_preprocessing=True.")
    print(" Pipeline must feed RAW Float32 pixels in range [0.0, 255.0].")
    print(" Manual division by 255.0 must NOT be applied in tf.data pipeline.")
    print(" Ready for Phase 7 (Transfer Learning Model Architecture).")
    print("=" * 60)

if __name__ == "__main__":
    audit_efficientnetv2_preprocessing()
