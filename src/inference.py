import os
import sys
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# Paths setup
BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path(r"d:\Skin Disease classification")
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

DISEASE_MODEL_PATH = MODELS_DIR / "disease_classifier.keras"
VALIDATOR_MODEL_PATH = MODELS_DIR / "image_validator.keras"
CLASS_MAP_JSON = OUTPUT_DIR / "class_mapping.json"
UNCERTAINTY_CONFIG_JSON = OUTPUT_DIR / "uncertainty_config.json"

# Load OpenCV Cascade Detectors for Face/Selfie Guardrail
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

class SkinDiseaseInferenceEngine:
    """
    Multi-Layered Production Guardrail Inference Engine for Skin Disease Classification.
    Combines:
    1. Gate 1a: Face & Facial Feature Detection Guardrail (Rejects Face Selfies)
    2. Gate 1b: MobileNetV2 OOD Domain Validator
    3. Main Model: EfficientNetV2B0 Disease Classifier
    4. Gate 2: Empirical Uncertainty Rejection Rule (Tau = 0.65)
    5. Explainability: Grad-CAM Heatmap Generation
    """
    def __init__(self):
        self.load_models_and_configs()

    def load_models_and_configs(self):
        if not DISEASE_MODEL_PATH.exists() or not VALIDATOR_MODEL_PATH.exists():
            raise FileNotFoundError("Trained models missing! Ensure Phase 10 & Phase 14 are completed.")

        # Load models
        self.disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)
        self.validator_model = tf.keras.models.load_model(VALIDATOR_MODEL_PATH)

        # Load mappings & configs
        with open(CLASS_MAP_JSON, 'r') as f:
            self.class_map = json.load(f)
        self.inv_class_map = {idx: name for name, idx in self.class_map.items()}
        self.class_names = [self.inv_class_map[i] for i in range(len(self.class_map))]

        with open(UNCERTAINTY_CONFIG_JSON, 'r') as f:
            self.uncertainty_config = json.load(f)

        self.validator_thresh = self.uncertainty_config.get("validator_threshold", 0.50)
        self.uncertainty_thresh = self.uncertainty_config.get("uncertainty_threshold", 0.65)

    def preprocess_image(self, img_input):
        """Processes PIL Image or file path into standard (224, 224, 3) float32 tensor and uint8 OpenCV image."""
        if isinstance(img_input, (str, Path)):
            img_np = cv2.imread(str(img_input))
            if img_np is None:
                raise ValueError(f"Unable to read image at {img_input}")
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        elif isinstance(img_input, Image.Image):
            img_rgb = np.array(img_input.convert('RGB'))
        elif isinstance(img_input, np.ndarray):
            img_rgb = img_input if img_input.shape[2] == 3 else cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError("Unsupported image input type!")

        img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        img_tensor = tf.cast(tf.convert_to_tensor(img_resized), tf.float32)
        return img_tensor, img_rgb

    def detect_face_selfie(self, img_rgb):
        """Gate 1a: Checks if the image contains a prominent human face selfie."""
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # Detect face with strict thresholds to prevent false positives on skin lesions & moles
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(60, 60))
        if len(faces) > 0:
            return True, "Human Face / Selfie detected. Please upload a close-up photo of the skin lesion only."

        return False, None

    def validate_image(self, img_tensor, img_rgb):
        """Gate 1: Combines Face Detection + MobileNetV2 OOD Domain Validator."""
        # Check Face/Selfie Guardrail
        is_face, face_msg = self.detect_face_selfie(img_rgb)
        if is_face:
            return False, 0.0, face_msg

        # Check OOD Validator Model
        img_batch = tf.expand_dims(img_tensor, 0)
        score = float(self.validator_model.predict(img_batch, verbose=0)[0][0])
        is_valid = score >= self.validator_thresh
        
        msg = "Image belongs to valid skin lesion domain." if is_valid else self.uncertainty_config.get("rejection_message_invalid")
        return is_valid, score, msg

    def generate_gradcam(self, img_tensor, pred_index):
        """Generates Grad-CAM Heatmap overlay for EfficientNetV2B0."""
        img_batch = tf.expand_dims(img_tensor, 0)
        
        base_model = None
        for layer in self.disease_model.layers:
            if "efficientnet" in layer.name.lower():
                base_model = layer
                break

        target_conv = None
        for layer in reversed(base_model.layers):
            if layer.name == "top_conv" or isinstance(layer, tf.keras.layers.Conv2D):
                target_conv = layer
                break

        grad_model = tf.keras.models.Model(
            inputs=[base_model.inputs],
            outputs=[target_conv.output, base_model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, base_outputs = grad_model(img_batch)
            x = base_outputs
            for layer in self.disease_model.layers:
                if layer.name in ["global_avg_pooling", "head_batch_norm", "head_dense_256", "head_dropout", "output_softmax"]:
                    x = layer(x)
            preds = x
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.maximum(tf.squeeze(heatmap), 0.0) / (tf.reduce_max(heatmap) + 1e-10)
        heatmap_np = heatmap.numpy()

        orig_img = img_tensor.numpy().astype("uint8")
        heatmap_resized = cv2.resize(heatmap_np, (224, 224))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(orig_img, 0.6, color_heatmap, 0.4, 0)

        return heatmap_np, color_heatmap, overlay

    def predict_image(self, img_input):
        """
        Full Production Dual-Gate Inference Pipeline.
        """
        img_tensor, img_rgb = self.preprocess_image(img_input)

        # GATE 1: Input Domain & Face Validation
        is_valid_domain, val_score, val_msg = self.validate_image(img_tensor, img_rgb)
        if not is_valid_domain:
            return {
                "status": "REJECTED_INVALID",
                "is_valid_domain": False,
                "validator_score": round(val_score, 4),
                "predicted_class_idx": None,
                "predicted_class_name": None,
                "confidence": 0.0,
                "all_probabilities": {},
                "message": val_msg,
                "gradcam_overlay": None
            }

        # GATE 2: Main Classifier Inference
        img_batch = tf.expand_dims(img_tensor, 0)
        probs = self.disease_model.predict(img_batch, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_name = self.inv_class_map[pred_idx]

        prob_dict = {self.inv_class_map[i]: round(float(probs[i]), 4) for i in range(len(probs))}

        # Uncertainty Threshold Check (Tau = 0.65)
        if confidence < self.uncertainty_thresh:
            _, _, overlay = self.generate_gradcam(img_tensor, pred_idx)
            return {
                "status": "REJECTED_UNCERTAIN",
                "is_valid_domain": True,
                "validator_score": round(val_score, 4),
                "predicted_class_idx": pred_idx,
                "predicted_class_name": pred_name,
                "confidence": round(confidence, 4),
                "all_probabilities": prob_dict,
                "message": self.uncertainty_config.get("rejection_message_uncertain"),
                "gradcam_overlay": overlay
            }

        # ACCEPTED Prediction
        _, _, overlay = self.generate_gradcam(img_tensor, pred_idx)
        return {
            "status": "ACCEPTED",
            "is_valid_domain": True,
            "validator_score": round(val_score, 4),
            "predicted_class_idx": pred_idx,
            "predicted_class_name": pred_name,
            "confidence": round(confidence, 4),
            "all_probabilities": prob_dict,
            "message": "Prediction accepted with high confidence.",
            "gradcam_overlay": overlay
        }
