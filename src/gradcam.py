import numpy as np
import cv2
import tensorflow as tf

def make_gradcam_heatmap(img_array, full_model, last_conv_layer_name="top_conv", pred_index=None):
    """
    Computes Grad-CAM heatmap for EfficientNetV2B0 model.
    """
    base_model = None
    for layer in full_model.layers:
        if "efficientnet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("EfficientNet base layer not found in model!")

    target_conv_layer = None
    for layer in reversed(base_model.layers):
        if layer.name == last_conv_layer_name or isinstance(layer, tf.keras.layers.Conv2D):
            target_conv_layer = layer
            break

    grad_model = tf.keras.models.Model(
        inputs=[base_model.inputs],
        outputs=[target_conv_layer.output, base_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = grad_model(img_array)
        x = base_outputs
        for layer in full_model.layers:
            if layer.name in ["global_avg_pooling", "head_batch_norm", "head_dense_256", "head_dropout", "output_softmax"]:
                x = layer(x)
        preds = x
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0.0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def generate_gradcam_overlay(img_tensor, heatmap):
    """Generates RGB JET overlay blending heatmap with original image."""
    orig_img = img_tensor.numpy().astype("uint8")
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(orig_img, 0.6, color_heatmap, 0.4, 0)
    return color_heatmap, overlay
