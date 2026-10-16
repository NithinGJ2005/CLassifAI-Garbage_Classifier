import json
import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    HAS_TF = True
except ImportError:
    HAS_TF = False
    def preprocess_input(arr):
        """Exact numerical equivalent of MobileNetV2 preprocess_input: maps [0, 255] to [-1, 1]."""
        return (arr / 127.5) - 1.0

# --------------------------------------------------
# CONFIG & PATHS
# --------------------------------------------------
st.set_page_config(
    page_title="ClassifAI - Garbage Classifier App",
    page_icon="♻️",
    layout="wide"
)

KERAS_MODEL_PATH = Path("artifacts/models/garbage_classifier.keras")
LEGACY_MODEL_PATH = Path("artifacts/models/garbage_classifier.h5")
CLASS_NAMES_PATH = Path("artifacts/models/class_names.json")
METRICS_PATH = Path("artifacts/models/test_metrics.json")
SDG_DIR = Path("assets/sdg")

DEFAULT_CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
DEFAULT_CONFIDENCE_THRESHOLD = 65.0

# --------------------------------------------------
# MODEL RESOLUTION & DATA UTILITIES
# --------------------------------------------------
def get_model_path():
    """Resolves model path adhering to hierarchy: .keras first, .h5 legacy fallback."""
    if KERAS_MODEL_PATH.exists():
        return KERAS_MODEL_PATH, "MobileNetV2 (.keras)"
    elif LEGACY_MODEL_PATH.exists():
        return LEGACY_MODEL_PATH, "MobileNetV2 Legacy (.h5)"
    return None, "Missing Model"

def load_class_names():
    """Loads class names from JSON if available, otherwise defaults to the six standard classes."""
    if CLASS_NAMES_PATH.exists():
        try:
            with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                names = json.load(f)
            if isinstance(names, list) and len(names) == 6:
                return [n.lower().strip() for n in names]
        except Exception:
            pass
    return DEFAULT_CLASS_NAMES

def prepare_image(pil_image):
    """
    Converts image to RGB, resizes to (224, 224), and applies mobilenet_v2.preprocess_input.
    Ensures pixel normalization to [-1, 1] is applied exactly once.
    """
    rgb_img = pil_image.convert("RGB")
    resized_img = rgb_img.resize((224, 224))
    img_arr = np.array(resized_img, dtype=np.float32)
    preprocessed = preprocess_input(img_arr)
    batch_tensor = np.expand_dims(preprocessed, axis=0)
    return batch_tensor

CLASS_NAMES = load_class_names()

# --------------------------------------------------
# STYLING (FULL DARK MODE + RESPONSIVE CARDS)
# --------------------------------------------------
st.markdown("""
<style>
/* 1. Global Background & Text */
.stApp {
    background: linear-gradient(180deg, #020617 0%, #020617 100%);
    color: #ffffff !important;
}

/* 2. REMOVE TOP WHITE GAP */
.block-container {
    padding-top: 1.8rem !important;
    margin-top: 0px !important;
}
header {
    background-color: rgba(0,0,0,0) !important;
}

/* 3. Global Text Visibility Force */
p, .stMarkdown, h1, h2, h3, h4, h5, h6, [data-testid="stCaptionContainer"] {
    color: #ffffff !important;
}

/* 4. FILE UPLOADER - DARK & VISIBLE */
[data-testid='stFileUploader'] {
    width: 100%;
}
[data-testid='stFileUploader'] section {
    background-color: #111827;
    border: 1px dashed #3b82f6;
}
[data-testid='stFileUploader'] * {
    color: #ffffff !important;
}
[data-testid='stFileUploader'] button {
    background-color: #1f2937 !important;
    color: #ffffff !important;
    border: 1px solid #374151 !important;
}
[data-testid='stFileUploader'] small {
    color: #cbd5e1 !important;
}
[data-testid='stFileUploader'] button[kind="secondary"] {
    color: #ffffff !important;
    border-color: #ffffff !important;
}

/* 5. Section Headers */
.section-header {
    background: #1e293b;
    padding: 10px 14px;
    border-radius: 6px;
    font-weight: 600;
    margin-bottom: 12px;
    border-left: 4px solid #3b82f6;
    font-size: 15px;
    display: flex;
    align-items: center;
    color: #ffffff;
}

/* 6. Confidence Card */
.prediction-card {
    background: #0f172a;
    padding: 20px;
    border-radius: 8px;
    margin-bottom: 12px;
    text-align: center;
    border: 1px solid #334155;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}
.muted-label {
    color: #cbd5e1 !important; 
    font-size: 12px;
    margin-top: 4px;
    font-weight: 500;
}

/* 7. Impact Bars */
.impact-box {
    background-color: rgba(6, 78, 59, 0.6);
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 8px;
    border-left: 4px solid #10b981;
    font-size: 14px;
    color: #ecfdf5 !important; 
}

/* 8. Warning Box */
.warning-box {
    background-color: rgba(120, 53, 15, 0.65);
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 12px;
    border-left: 4px solid #f59e0b;
    font-size: 13px;
    color: #fef3c7 !important;
}

/* 9. Badge */
.status-badge {
    display: inline-block;
    padding: 4px 10px;
    background-color: #1e293b;
    color: #93c5fd;
    font-size: 12px;
    font-weight: 600;
    border-radius: 9999px;
    border: 1px solid #3b82f6;
    margin-top: 4px;
    margin-bottom: 14px;
}

/* 10. Button Styling */
div.stButton > button {
    width: 100%;
    background-color: #3b82f6;
    color: white;
    font-weight: bold;
    border: none;
    padding: 10px;
    border-radius: 6px;
    margin-top: 10px;
}
div.stButton > button:hover {
    background-color: #2563eb;
    color: white;
}

/* 11. SIDEBAR DARK STYLING (Complete Coverage & High Contrast) */
[data-testid="stSidebar"], 
section[data-testid="stSidebar"], 
[data-testid="stSidebarContent"],
[data-testid="stSidebarUserContent"],
.stSidebar {
    background: #090d16 !important;
    background-color: #090d16 !important;
    border-right: 1px solid #1e293b !important;
}

/* Ensure nested sidebar divs inherit the dark background */
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] > div,
[data-testid="stSidebarUserContent"] > div {
    background-color: #090d16 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] strong {
    color: #ffffff !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] li {
    color: #cbd5e1 !important;
}

/* All code badges everywhere and in sidebar */
code, 
[data-testid="stSidebar"] code,
.stCode {
    background-color: #1e293b !important;
    color: #38bdf8 !important;
    border: 1px solid #334155 !important;
    padding: 2px 7px !important;
    border-radius: 4px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

/* Sidebar horizontal rule */
[data-testid="stSidebar"] hr {
    border-color: #1e293b !important;
    margin: 16px 0 !important;
}

/* Sidebar slider label, thumb, and values */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stSlider div[data-testid="stMarkdownContainer"] p {
    color: #93c5fd !important;
}

/* Sidebar expander */
[data-testid="stSidebar"] details {
    background-color: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] details summary {
    color: #f1f5f9 !important;
    font-weight: 600 !important;
}

/* Sidebar collapse/expand controls */
[data-testid="stSidebarCollapseButton"] button, 
[data-testid="stSidebarCollapsedControl"] button,
button[aria-label="Close sidebar"],
button[aria-label="Open sidebar"] {
    color: #94a3b8 !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SUSTAINABLE DEVELOPMENT GOALS & CARBON DATA
# --------------------------------------------------
SDG_MAPPING = {
    "plastic": [6, 12, 14],
    "paper": [12, 13, 15],
    "glass": [12, 13],
    "metal": [7, 12, 13],
    "cardboard": [12, 15],
    "trash": [7, 12],
}

IMPACT_MAPPING = {
    "cardboard": [
        "📦 Cardboard Waste:",
        "• Virgin cardboard: emits approx. 0.5–0.9 kg CO₂/kg.",
        "• Recycled cardboard: 0.2–0.4 kg CO₂/kg.",
        "♻ Recycling reduces water, energy use, and deforestation."
    ],
    "plastic": [
        "🧴 Plastic Waste:",
        "• PET (bottles): 2.1–3.5 kg CO₂/kg.",
        "• HDPE (containers): 1.7–2.8 kg CO₂/kg.",
        "• General plastic avg: 3–6 kg CO₂/kg.",
        "♻ Recycling plastic saves up to 30–60% emissions."
    ],
    "glass": [
        "🍾 Glass Waste:",
        "• New glass: 0.6–1.2 kg CO₂/kg due to melting.",
        "• Recycled glass: 0.2–0.4 kg CO₂/kg.",
        "♻ Glass is 100% recyclable without quality loss."
    ],
    "metal": [
        "🛠 Metal Waste:",
        "• Aluminum: 10–13 kg CO₂/kg (virgin).",
        "• Aluminum (Recycled): 0.6–1 kg CO₂/kg.",
        "• Steel: 1.8–2.5 kg CO₂/kg (60% less if recycled).",
        "♻ Recycling metals saves significant energy."
    ],
    "paper": [
        "📄 Paper Waste:",
        "• Virgin paper: 1.8–2.5 kg CO₂/kg.",
        "• Recycled paper: 0.9–1.5 kg CO₂/kg.",
        "♻ Recycling paper reduces landfill methane & preserves forests."
    ],
    "trash": [
        "🗑 Mixed General Trash:",
        "• Emissions vary widely depending on materials.",
        "• Can exceed 5 kg CO₂/kg (esp. with food/plastic).",
        "♻ Waste separation reduces footprint significantly."
    ]
}

# --------------------------------------------------
# LOAD MODEL (CACHED)
# --------------------------------------------------
active_model_path, active_model_type = get_model_path()

@st.cache_resource
def load_classifier_model(path_str):
    if not path_str or not HAS_TF:
        return None
    return tf.keras.models.load_model(path_str)

model = None
if HAS_TF and active_model_path:
    try:
        model = load_classifier_model(str(active_model_path))
    except Exception as e:
        st.error(f"Error loading model from {active_model_path}: {e}")

# --------------------------------------------------
# SIDEBAR CONFIGURATION
# --------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Classification Settings")
    confidence_threshold = st.slider(
        "Low Confidence Warning Threshold (%)",
        min_value=40.0,
        max_value=90.0,
        value=DEFAULT_CONFIDENCE_THRESHOLD,
        step=5.0,
        help="Alerts the user when model confidence is low, indicating uncertain predictions."
    )
    st.markdown("---")
    st.markdown("### ℹ️ Active Model Information")
    st.markdown(f"**Loaded Architecture:** `{active_model_type}`")
    st.markdown(f"**Source File:** `{active_model_path.name if active_model_path else 'None'}`")
    st.markdown(f"**Target Classes ({len(CLASS_NAMES)}):**")
    for i, c in enumerate(CLASS_NAMES):
        st.markdown(f"  {i+1}. `{c}`")
    
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                m_data = json.load(f)
            st.markdown("---")
            with st.expander("📊 Held-Out Test Evaluation", expanded=False):
                acc_val = m_data.get("overall_accuracy", 0) * 100
                f1_val = m_data.get("macro_f1", 0)
                lat_val = m_data.get("average_latency_ms_per_image", 0)
                st.metric("Test Accuracy", f"{acc_val:.2f}%")
                st.metric("Macro F1-Score", f"{f1_val:.4f}")
                st.metric("Inference Latency", f"{lat_val:.2f} ms")
                st.caption(f"Architecture: {m_data.get('selected_model', 'Trained Model')} evaluated on 4,478 unseen test images.")
        except Exception:
            pass

    st.markdown("---")
    st.caption("ClassifAI uses MobileNetV2 with transfer learning and ImageNet weights.")

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("♻️ ClassifAI - Garbage Classifier App")
st.markdown(f'<span class="status-badge">Model Status: {active_model_type}</span>', unsafe_allow_html=True)
st.markdown("Upload an image of waste and click **Classify Waste** below to identify its material category and review its environmental footprint.")

if not HAS_TF:
    st.warning(
        "⏳ **TensorFlow is not installed in the local environment.**\n\n"
        f"The model file `{active_model_path.name if active_model_path else 'model'}` is present in `artifacts/models/`, "
        "but TensorFlow is required to run inference locally.\n\n"
        "Please run `pip install -r requirements.txt` (or wait for the pip installation to complete) and then refresh this page."
    )
elif not model:
    st.error(
        "⚠️ **No trained model found!**\n\n"
        "Please place `garbage_classifier.keras` (or fallback `garbage_classifier.h5`) into the "
        "`artifacts/models/` directory.\n\n"
        "Follow the instructions in `TRAINING_IN_COLAB.md` to train and export the model using Google Colab."
    )

uploaded_file = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"], key="uploader")

# --------------------------------------------------
# MAIN CLASSIFICATION LOGIC
# --------------------------------------------------
if uploaded_file and model:
    try:
        image = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Could not read uploaded image: {e}")
        st.stop()

    col1, col2, col3 = st.columns([1, 1, 1])

    # --------------------------------------------------
    # LEFT: UPLOADED IMAGE & CLASSIFY BUTTON
    # --------------------------------------------------
    with col1:
        st.markdown('<div class="section-header">Uploaded Image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        classify_btn = st.button("Classify Waste")

    # --------------------------------------------------
    # RUN PREDICTION ON BUTTON CLICK
    # --------------------------------------------------
    if classify_btn:
        # Preprocess input (exactly once, converting to RGB and scaling to [-1, 1])
        batch_tensor = prepare_image(image)

        # Predict
        preds = model.predict(batch_tensor, verbose=0)
        idx = int(np.argmax(preds[0]))
        prediction = CLASS_NAMES[idx]
        confidence = float(preds[0][idx] * 100)

        sdgs = SDG_MAPPING.get(prediction, [])
        impact = IMPACT_MAPPING.get(prediction, [])
        is_low_confidence = confidence < confidence_threshold

        # --------------------------------------------------
        # MIDDLE: RESULT & SDGs
        # --------------------------------------------------
        with col2:
            st.markdown('<div class="section-header">Classification Result</div>', unsafe_allow_html=True)

            # Confidence Card
            st.markdown(f"""
            <div class="prediction-card">
                <div style="color:#facc15; font-size:24px; font-weight:800; text-transform:uppercase; letter-spacing:1px;">
                    {prediction}
                </div>
                <div class="muted-label">Statistical Model Confidence</div>
                <div style="color:#4ade80; font-size:32px; font-weight:800;">
                    {confidence:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Low Confidence Warning Alert
            if is_low_confidence:
                st.markdown(f"""
                <div class="warning-box">
                    ⚠️ <b>Low Confidence Prediction ({confidence:.1f}% &lt; {confidence_threshold:.0f}% threshold)</b><br>
                    The model is uncertain about this classification. The item might be partially obscured, 
                    contain mixed materials, or have unusual lighting. Please verify manually.
                </div>
                """, unsafe_allow_html=True)

            # SDG Icons
            if sdgs:
                st.markdown('<div style="font-weight:600; color:#ffffff; margin-bottom:10px;">Related Sustainable Development Goals</div>', unsafe_allow_html=True)
                cols = st.columns(len(sdgs))
                for c, num in zip(cols, sdgs):
                    with c:
                        sdg_icon_path = SDG_DIR / f"sdg_{num}.png"
                        if sdg_icon_path.exists():
                            st.image(str(sdg_icon_path), use_container_width=True)
                        else:
                            st.caption(f"SDG {num}")

            # Validation Box
            st.markdown(f"""
            <div class="impact-box" style="margin-top:12px; background:#064e3b; border-color:#34d399;">
                ✅ The image is classified as <b>{prediction.capitalize()}</b>
            </div>
            """, unsafe_allow_html=True)

        # --------------------------------------------------
        # RIGHT: CARBON FOOTPRINT AWARENESS
        # --------------------------------------------------
        with col3:
            st.markdown('<div class="section-header">Carbon Footprint Awareness</div>', unsafe_allow_html=True)
            for line in impact:
                st.markdown(f'<div class="impact-box">{line}</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption(
    "Disclaimer: Classification results reflect statistical probabilities generated by MobileNetV2 and are not guaranteed "
    "to be 100% correct. SDG indicators and carbon estimates are provided for environmental education and awareness."
)