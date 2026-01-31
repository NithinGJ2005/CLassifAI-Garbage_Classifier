import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Garbage Classifier App",
    layout="wide"
)

MODEL_PATH = Path("artifacts/models/garbage_classifier.h5")
SDG_DIR = Path("assets/sdg")

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# --------------------------------------------------
# STYLING (FULL DARK MODE + NO TOP GAP)
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
    padding-top: 2rem !important;
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

/* 8. NEW: Button Styling (To match the theme) */
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
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# DATA
# --------------------------------------------------
SDG_MAPPING = {
    "plastic": [6, 12, 14],
    "paper": [12, 13, 15],
    "glass": [12, 13],
    "metal": [7, 12, 13],
    "cardboard": [12, 15],
    "trash": [7, 12],
}

# UPDATED CARBON INFO (Formatted as lists for the UI)
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
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("♻️ ClassifAI - A Garbage Classifier App")
st.markdown("Upload a image of the waste and click the Classify Waste button below to identify its type and understand its environmental impact.")

uploaded_file = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"], key="uploader")

# --------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    
    # Define columns immediately so we can place the button in Col 1
    col1, col2, col3 = st.columns([1, 1, 1])

    # --------------------------------------------------
    # LEFT: UPLOADED IMAGE & BUTTON
    # --------------------------------------------------
    with col1:
        st.markdown('<div class="section-header">Uploaded Image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        
        # --- NEW CLASSIFY BUTTON ---
        classify_btn = st.button("Classify Waste")

    # --------------------------------------------------
    # LOGIC: ONLY RUNS IF BUTTON CLICKED
    # --------------------------------------------------
    if classify_btn:
        
        # Preprocess
        img = image.resize((224, 224))
        img_arr = np.array(img)
        img_arr = preprocess_input(img_arr)
        img_arr = np.expand_dims(img_arr, axis=0)

        # Predict
        preds = model.predict(img_arr)
        idx = np.argmax(preds)
        prediction = CLASS_NAMES[idx]
        confidence = preds[0][idx] * 100

        sdgs = SDG_MAPPING.get(prediction, [])
        impact = IMPACT_MAPPING.get(prediction, [])

        # --------------------------------------------------
        # MIDDLE: RESULT (Top) -> SDGs (Bottom)
        # --------------------------------------------------
        with col2:
            st.markdown('<div class="section-header">Classification Result</div>', unsafe_allow_html=True)

            # 1. THE CONFIDENCE CARD
            st.markdown(f"""
            <div class="prediction-card">
                <div style="color:#facc15; font-size:24px; font-weight:800; text-transform:uppercase; letter-spacing:1px;">
                    {prediction}
                </div>
                <div class="muted-label">Confidence Score</div>
                <div style="color:#4ade80; font-size:32px; font-weight:800;">
                    {confidence:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 2. SDGs Header + Icons
            if sdgs:
                st.markdown('<div style="font-weight:600; color:#ffffff; margin-bottom:10px;"> Related Sustainable Development Goals</div>', unsafe_allow_html=True)
                
                cols = st.columns(len(sdgs))
                for c, num in zip(cols, sdgs):
                    with c:
                        st.image(str(SDG_DIR / f"sdg_{num}.png"), use_container_width=True)

            # 3. Validation Box
            st.markdown(f"""
            <div class="impact-box" style="margin-top:12px; background:#064e3b; border-color:#34d399;">
                ✅ The image is classified as <b>{prediction.capitalize()}</b>
            </div>
            """, unsafe_allow_html=True)

        # --------------------------------------------------
        # RIGHT: CARBON FOOTPRINT
        # --------------------------------------------------
        with col3:
            st.markdown('<div class="section-header">Carbon Footprint Awareness</div>', unsafe_allow_html=True)
            for line in impact:
                st.markdown(f'<div class="impact-box">{line}</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("SDG (Sustainable Development Goals) icons and environmental insights are displayed to promote sustainable waste management awareness.")