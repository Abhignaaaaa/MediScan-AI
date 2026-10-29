"""
app.py - Streamlit web interface for MediScan AI.

MediScan AI is an educational medical image analysis assistant demonstrating
computer vision inference with the TorchXRayVision DenseNet-121 model.
"""

from __future__ import annotations

import io
import os
from typing import Optional
import streamlit as st
from PIL import Image

from image_utils import (
    validate_image_file,
    preprocess_chest_xray,
    ImageValidationError,
    SUPPORTED_EXTENSIONS,
    MAX_IMAGE_SIZE_BYTES,
)
from model_utils import (
    load_model,
    run_inference,
    ModelInferenceError,
    MODEL_NAME,
)

# Page configuration
st.set_page_config(
    page_title="MediScan AI — Medical Image Analysis Assistant",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Blue-and-White Medical Research Theme)
st.markdown(
    """
    <style>
    /* Global brand styling */
    .main-title {
        color: #0b3c61;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #2b6cb0;
        font-size: 1.15rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #ebf8ff;
        border-left: 5px solid #3182ce;
        padding: 1rem 1.2rem;
        border-radius: 4px;
        margin: 1.2rem 0;
        color: #2c5282;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .disclaimer-box strong {
        color: #1a365d;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-elevated {
        background-color: #feebc8;
        color: #9c4221;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #fbd38d;
    }
    .badge-nominal {
        background-color: #edf2f7;
        color: #4a5568;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .section-header {
        color: #1a365d;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing pretrained TorchXRayVision DenseNet-121 model...")
def get_cached_model():
    """Caches model loading to ensure instant subsequent inferences."""
    return load_model()


def render_header():
    """Renders application header and branding."""
    st.markdown('<div class="main-title">🫁 MediScan AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Educational Medical Image Analysis Assistant · Computer Vision Demonstration</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="disclaimer-box">
            <strong>⚠️ Educational-Use Disclaimer:</strong><br>
            MediScan AI is an educational demonstration of pretrained computer vision models.
            Its outputs are <strong>not medical diagnoses</strong> and must not be used for treatment decisions.
            Model performance may vary depending on image quality, patient population, acquisition conditions,
            and the model's training data. Consult a qualified healthcare professional for medical interpretation.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Renders educational sidebar with model specifications and references."""
    with st.sidebar:
        st.header("Model Specifications")
        st.markdown(f"**Architecture:** DenseNet-121")
        st.markdown(f"**Weights Key:** `{MODEL_NAME}`")
        st.markdown(f"**Library:** TorchXRayVision v1.5.7")
        st.markdown(f"**License:** Apache License 2.0")
        st.markdown(f"**Input Specs:** Grayscale, 224×224 px")
        st.markdown(f"**Pathologies Evaluated:** 18 Findings")

        st.markdown("---")
        st.header("Training Datasets")
        st.markdown(
            """
            Multi-center radiological datasets:
            - **NIH ChestX-ray14**
            - **CheXpert** (Stanford)
            - **MIMIC-CXR** (Beth Israel Deaconess)
            - **PadChest** (BIMCV-CSUSP)
            - **Google OpenI / Kaggle RSNA**
            """
        )

        st.markdown("---")
        st.header("Interpretation Guide")
        st.markdown(
            """
            - **Normalized Model Score (0.0 to 1.0):** Output rescaled by TorchXRayVision `op_norm` so that the empirical operating threshold $\\theta$ maps to $0.50$.
            - **Score ≥ 0.50:** The model output meets or exceeds the empirical operating point threshold for this class.
            - **Score < 0.50:** The model output is below the empirical operating point threshold.
            - **Raw Sigmoid Output (0.0 to 1.0):** Direct uncalibrated logistic sigmoid activation $\\sigma(z) = \\frac{1}{1 + e^{-z}}$.
            - **⚠️ Key Limitation:** Neither score represents a calibrated disease probability. They are mathematical activations reflecting learned radiographic patterns, not confirmed diagnoses or clinical likelihoods.
            """
        )


def main():
    render_header()
    render_sidebar()

    # Load model with friendly error handling
    try:
        model = get_cached_model()
    except ModelInferenceError as e:
        st.error(f"❌ Model Initialization Error: {str(e)}")
        st.info("Please verify internet access to download pretrained weights or check local cache.")
        return

    col_upload, col_preview = st.columns([1, 1], gap="large")

    uploaded_bytes: Optional[bytes] = None
    uploaded_name: str = "uploaded_scan.png"

    with col_upload:
        st.subheader("1. Select or Upload Image")
        st.markdown(
            f"Upload a frontal chest X-ray image (PA or AP projection). "
            f"Supported formats: **{', '.join(sorted(SUPPORTED_EXTENSIONS)).upper()}** "
            f"(Max size: {MAX_IMAGE_SIZE_BYTES // (1024*1024)} MB)."
        )

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image file",
            type=list(SUPPORTED_EXTENSIONS),
            help="Upload a standard PNG or JPEG chest radiograph.",
        )

        # Sample image loader
        st.markdown("**— OR try a legal public-access sample image —**")
        sample_path = os.path.join("samples", "nih_sample_00000001_000.png")
        use_sample = st.button("📥 Load NIH ChestX-ray14 Educational Sample")

        if use_sample and os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                uploaded_bytes = f.read()
            uploaded_name = "nih_sample_00000001_000.png"
            st.session_state["active_image_bytes"] = uploaded_bytes
            st.session_state["active_image_name"] = uploaded_name
            st.success("Loaded verified public-domain NIH sample image.")
        elif uploaded_file is not None:
            uploaded_bytes = uploaded_file.getvalue()
            uploaded_name = uploaded_file.name
            st.session_state["active_image_bytes"] = uploaded_bytes
            st.session_state["active_image_name"] = uploaded_name
        elif "active_image_bytes" in st.session_state:
            uploaded_bytes = st.session_state["active_image_bytes"]
            uploaded_name = st.session_state["active_image_name"]

    # Image validation & preview
    validated_image: Optional[Image.Image] = None
    image_metadata = None

    if uploaded_bytes is not None:
        try:
            validated_image, image_metadata = validate_image_file(uploaded_bytes, uploaded_name)
        except ImageValidationError as err:
            st.error(f"❌ Image Validation Failed: {str(err)}")
            return

    with col_preview:
        st.subheader("2. Image Preview & Validation")
        if validated_image is not None and image_metadata is not None:
            st.image(validated_image, caption=f"Preview: {uploaded_name}", use_container_width=True)
            st.caption(
                f"**Format:** {image_metadata['format']} | "
                f"**Dimensions:** {image_metadata['width']} × {image_metadata['height']} px | "
                f"**Size:** {image_metadata['size_bytes'] / 1024:.1f} KB"
            )
        else:
            st.info("👈 Please upload an image or load the educational sample to preview.")

    # Analysis Action
    if validated_image is not None:
        st.markdown("---")
        st.subheader("3. Pretrained Model Analysis")

        analyze_btn = st.button("🔍 Analyze Radiograph with Pretrained DenseNet-121", type="primary")

        if analyze_btn or "last_analysis" in st.session_state:
            with st.spinner("Executing preprocessing and TorchXRayVision neural inference..."):
                try:
                    # Preprocess according to model specification
                    input_tensor = preprocess_chest_xray(validated_image)
                    # Run genuine inference
                    analysis_results = run_inference(model, input_tensor)
                    st.session_state["last_analysis"] = analysis_results
                except (ImageValidationError, ModelInferenceError) as err:
                    st.error(f"❌ Analysis Failed: {str(err)}")
                    return

            results = st.session_state["last_analysis"]

            # Display Results Section
            st.markdown('<div class="section-header"><h4>Model Findings & Radiographic Patterns</h4></div>', unsafe_allow_html=True)

            elevated_count = results["elevated_count"]
            all_findings = results["all_findings"]

            if elevated_count > 0:
                st.warning(
                    f"⚠️ The model identified **{elevated_count} radiographic pattern(s)** "
                    f"meeting or exceeding the empirical operating threshold (score ≥ 0.50)."
                )
            else:
                st.success(
                    "✅ No radiographic findings met or exceeded the empirical operating threshold "
                    "(all scores < 0.50)."
                )

            # Three tabs: Findings Exceeding Threshold, Complete Breakdown, and Technical Details
            tab_summary, tab_all, tab_tech = st.tabs([
                "📊 Findings Exceeding Threshold",
                "📋 All 18 Evaluated Pathologies",
                "🔬 Technical & Preprocessing Details"
            ])

            with tab_summary:
                if elevated_count == 0:
                    st.info("No findings exceeded the empirical operating threshold on this radiograph.")
                else:
                    for f in results["elevated_findings"]:
                        raw_sig = f.get("raw_sigmoid", f.get("raw_probability", 0.0))
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <h4 style="margin: 0; color: #1a365d;">{f['pathology']}</h4>
                                    <span class="badge-elevated">Normalized Score: {f['normalized_score']:.2f}</span>
                                </div>
                                <p style="margin: 0.5rem 0; color: #4a5568; font-size: 0.95rem;">{f['description']}</p>
                                <div style="font-size: 0.85rem; color: #718096;">
                                    <strong>Operating Threshold (θ):</strong> {f['operating_threshold']} | 
                                    <strong>Raw Sigmoid Output:</strong> {raw_sig:.4f}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            with tab_all:
                st.markdown(
                    "Complete breakdown of all 18 evaluated radiological conditions. "
                    "*Scores are uncalibrated model outputs and do not represent clinical disease probabilities.*"
                )
                table_data = []
                for item in all_findings:
                    raw_sig = item.get("raw_sigmoid", item.get("raw_probability", 0.0))
                    table_data.append({
                        "Radiographic Pathology": item["pathology"],
                        "Normalized Model Score": f"{item['normalized_score']:.4f}",
                        "Threshold Status": "⚠️ Exceeds Threshold (≥ 0.50)" if item.get("is_above_threshold", item.get("is_elevated", False)) else "Below Threshold (< 0.50)",
                        "Raw Sigmoid Output": f"{raw_sig:.4f}",
                        "Operating Threshold (θ)": f"{item['operating_threshold']:.4f}",
                        "Pathology Description": item["description"],
                    })
                st.dataframe(table_data, use_container_width=True, hide_index=True)

            with tab_tech:
                st.markdown("### Preprocessing Pipeline Applied")
                st.markdown(
                    """
                    1. **Format Validation:** Verified file header, MIME type, dimensions (32px to 8192px).
                    2. **Grayscale Conversion:** In-memory transformation to single 8-bit intensity channel (`L` mode).
                    3. **Intensity Normalization:** Intensity mapped from $[0, 255]$ to $[-1024, 1024]$ via $x_{norm} = \\frac{x}{255} \\times 2048 - 1024$.
                    4. **Aspect Ratio Preservation:** Spatial center-crop and bilinear resize to $224 \\times 224$ pixels.
                    5. **Tensor Formatting:** Float32 4D batch tensor of shape `(1, 1, 224, 224)`.
                    6. **Forward Pass:** PyTorch `model.eval()` with `torch.no_grad()`.
                    """
                )

            # Option to reset
            if st.button("🔄 Analyze Another Image"):
                for key in ["active_image_bytes", "active_image_name", "last_analysis"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()


if __name__ == "__main__":
    main()
