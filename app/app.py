import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path(r"d:\Skin Disease classification")
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.inference import SkinDiseaseInferenceEngine

# Page configuration
st.set_page_config(
    page_title="Skin Disease AI Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_inference_engine():
    """Cache inference engine so models are loaded only once."""
    return SkinDiseaseInferenceEngine()

def main():
    # Sidebar
    st.sidebar.title("🔬 Navigation & Info")
    st.sidebar.info(
        "**Architecture:** EfficientNetV2B0\n\n"
        "**Input Size:** 224 x 224 x 3\n\n"
        "**Guardrails:** MobileNetV2 OOD Validator + Uncertainty Threshold (Tau=0.65)\n\n"
        "**Explainability:** Grad-CAM Heatmap"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Supported Classes (9)")
    classes_list = [
        "Actinic keratosis", "Atopic Dermatitis", "Benign keratosis",
        "Dermatofibroma", "Melanocytic nevus", "Melanoma",
        "Squamous cell carcinoma", "Tinea Ringworm Candidiasis", "Vascular lesion"
    ]
    for c in classes_list:
        st.sidebar.markdown(f"- `{c}`")

    # Main Header
    st.title("🩺 AI-Based Skin Disease Image Classification System")
    st.markdown("##### Multi-Class Skin Lesion Analysis Powered by EfficientNetV2B0, Transfer Learning, Grad-CAM, and OOD Rejection")
    
    # Prominent Medical Disclaimer Alert Banner
    st.warning(
        "⚠️ **MEDICAL DISCLAIMER:** "
        "This application is an educational/research project and is NOT a medical diagnostic tool. "
        "Predictions may be incorrect. Always consult a qualified healthcare professional or dermatologist for any medical concerns."
    )

    st.markdown("---")

    # Image Uploader Section
    col_upload, col_preview = st.columns([1, 1])

    with col_upload:
        st.subheader("1. Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a skin image (JPG, PNG, JPEG)...",
            type=["jpg", "jpeg", "png"]
        )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            with col_preview:
                st.subheader("2. Uploaded Preview")
                st.image(image, caption="Uploaded Image", use_container_width=True)

            st.markdown("---")
            st.subheader("3. Diagnostic Analysis & Guardrail Validation")

            # Load Inference Engine
            with st.spinner("Initializing AI Models & Running Dual-Gate Guardrails..."):
                engine = get_inference_engine()
                result = engine.predict_image(image)

            # STEP 1: Domain Validator Check (Gate 1)
            if result["status"] == "REJECTED_INVALID":
                st.error("❌ **INPUT REJECTED (Out-Of-Domain Input)**")
                st.error(f"**Message:** {result['message']}")
                st.info(
                    "ℹ️ **Why was this image rejected?**\n"
                    "The binary image validator detected that this upload is outside the supported dermoscopic skin lesion domain "
                    "(e.g. selfie, face photo, animal, vehicle, landscape, screenshot, or object)."
                )
                st.stop() # STOP pipeline execution immediately!

            # STEP 2: Uncertainty Check (Gate 2)
            if result["status"] == "REJECTED_UNCERTAIN":
                st.warning("⚠️ **PREDICTION REJECTED (Low Confidence / High Uncertainty)**")
                st.warning(f"**Message:** {result['message']}")
                st.write(f"**Top Candidate:** `{result['predicted_class_name']}` (Confidence: `{result['confidence']*100:.1f}%` < Threshold `65.0%`)")
                st.info("ℹ️ The AI model detected ambiguity in the image features and refuses to issue a confident diagnosis.")
                
                # Show Probabilities Bar Chart
                st.subheader("Class Probability Breakdown")
                df_probs = pd.DataFrame(list(result["all_probabilities"].items()), columns=["Class", "Probability"])
                st.bar_chart(df_probs.set_index("Class"))
                st.stop()

            # STEP 3: ACCEPTED PREDICTION OUTPUT
            st.success("✅ **VALID INPUT ACCEPTED & CLASSIFIED**")
            
            # Display Prediction Metrics Cards
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Predicted Skin Disease", result["predicted_class_name"])
            with m_col2:
                st.metric("Model Confidence", f"{result['confidence']*100:.2f}%")
            with m_col3:
                st.metric("Validator Domain Score", f"{result['validator_score']*100:.1f}%")

            st.markdown("---")

            # Results Display Columns: Grad-CAM vs Class Probabilities
            res_col1, res_col2 = st.columns([1.2, 1])

            with res_col1:
                st.subheader("Grad-CAM Explainability Heatmap")
                st.caption("Red/Yellow regions indicate high visual feature attention influencing the model's prediction.")
                if result["gradcam_overlay"] is not None:
                    st.image(result["gradcam_overlay"], caption=f"Grad-CAM Overlay: {result['predicted_class_name']}", use_container_width=True)
                else:
                    st.info("Grad-CAM overlay unavailable.")

            with res_col2:
                st.subheader("All Class Probabilities")
                df_probs = pd.DataFrame(list(result["all_probabilities"].items()), columns=["Class", "Probability"])
                df_probs["Probability (%)"] = df_probs["Probability"] * 100
                st.dataframe(
                    df_probs[["Class", "Probability (%)"]].sort_values(by="Probability (%)", ascending=False),
                    use_container_width=True,
                    hide_index=True
                )
                st.bar_chart(df_probs.set_index("Class")["Probability (%)"])

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    else:
        st.info("👆 Please upload a skin lesion image above to get started.")

    # Footer Disclaimer
    st.markdown("---")
    st.caption(
        "**Educational/Research Project Disclaimer:** "
        "This AI application is designed strictly for educational and research demonstration purposes. "
        "It is not certified for clinical diagnostic use."
    )

if __name__ == "__main__":
    main()
