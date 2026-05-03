import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="MalariaAI — Detection System",
    page_icon="🔬",
    layout="centered"
)

# -----------------------------------
# CUSTOM CSS — Dark Medical Aesthetic
# -----------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* Base Reset */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #060b14;
    color: #e2eaf5;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 760px; }

/* ── HERO BANNER ── */
.hero {
    position: relative;
    text-align: center;
    padding: 3rem 2rem 2.5rem;
    margin-bottom: 2rem;
    background: linear-gradient(135deg, #0d1b2e 0%, #091423 60%, #040c18 100%);
    border: 1px solid #1b3050;
    border-radius: 20px;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse at 20% 50%, rgba(0,180,255,0.07) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(255,50,90,0.06) 0%, transparent 55%);
    pointer-events: none;
}
.hero-badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #00b4ff;
    background: rgba(0,180,255,0.1);
    border: 1px solid rgba(0,180,255,0.25);
    border-radius: 100px;
    padding: 0.3rem 1rem;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.1;
    margin: 0 0 0.6rem;
    background: linear-gradient(135deg, #ffffff 30%, #7ec8f5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1rem;
    color: #7090b0;
    margin: 0;
    font-weight: 400;
}

/* ── GRID STATS ── */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.stat-card {
    background: #0c1824;
    border: 1px solid #1a2e44;
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.stat-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #00b4ff;
    line-height: 1;
    margin-bottom: 0.35rem;
}
.stat-label {
    font-size: 0.72rem;
    color: #4a6a88;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── UPLOAD ZONE ── */
.upload-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #4a6a88;
    margin-bottom: 0.6rem;
}

/* Streamlit uploader override */
[data-testid="stFileUploader"] {
    background: #0c1824;
    border: 2px dashed #1a3a58 !important;
    border-radius: 16px !important;
    transition: border-color 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #00b4ff !important;
}
[data-testid="stFileUploader"] label {
    color: #7090b0 !important;
    font-family: 'Syne', sans-serif !important;
}

/* ── IMAGE PREVIEW ── */
.preview-wrap {
    border: 1px solid #1a3050;
    border-radius: 16px;
    overflow: hidden;
    margin: 1.5rem 0;
    position: relative;
}
.preview-tag {
    position: absolute;
    top: 10px; left: 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #00b4ff;
    background: rgba(0,20,40,0.85);
    border: 1px solid rgba(0,180,255,0.3);
    border-radius: 6px;
    padding: 3px 8px;
}

/* ── RESULT CARDS ── */
.result-card {
    border-radius: 18px;
    padding: 2rem;
    margin: 1.5rem 0 0.8rem;
    position: relative;
    overflow: hidden;
}
.result-card.danger {
    background: linear-gradient(135deg, #1a0812 0%, #250d1a 100%);
    border: 1px solid #5c1a30;
}
.result-card.safe {
    background: linear-gradient(135deg, #061a14 0%, #0d2a1e 100%);
    border: 1px solid #1a5c3a;
}
.result-card::after {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 120px; height: 120px;
    border-radius: 50%;
    opacity: 0.15;
}
.result-card.danger::after { background: #ff2255; }
.result-card.safe::after   { background: #00e87a; }

.result-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.result-status {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.result-card.danger .result-status { color: #ff6680; }
.result-card.safe  .result-status  { color: #00e87a; }
.result-label {
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.1;
}
.result-card.danger .result-label { color: #fff0f3; }
.result-card.safe  .result-label  { color: #f0fff8; }
.result-hint {
    font-size: 0.85rem;
    color: #6a8a7a;
    margin-top: 0.5rem;
}
.result-card.danger .result-hint { color: #8a6a72; }

/* ── CONFIDENCE BAR ── */
.conf-wrap {
    background: #0c1824;
    border: 1px solid #1a2e44;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.5rem;
}
.conf-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.8rem;
}
.conf-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4a6a88;
}
.conf-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #e2eaf5;
}
.conf-track {
    background: #0a1520;
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
    border: 1px solid #1a2e44;
}
.conf-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.8s ease;
}
.conf-fill.danger { background: linear-gradient(90deg, #c01840, #ff2255); }
.conf-fill.safe   { background: linear-gradient(90deg, #00a854, #00e87a); }

/* ── BREAKDOWN TABLE ── */
.breakdown {
    background: #0c1824;
    border: 1px solid #1a2e44;
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 2rem;
}
.breakdown-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4a6a88;
    padding: 0.8rem 1.4rem;
    border-bottom: 1px solid #1a2e44;
    background: #091320;
}
.breakdown-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.8rem 1.4rem;
    border-bottom: 1px solid #0f1e30;
    font-size: 0.9rem;
}
.breakdown-row:last-child { border-bottom: none; }
.breakdown-key { color: #7090b0; font-size: 0.85rem; }
.breakdown-val {
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: #c8daf0;
}

/* ── FOOTER ── */
.footer {
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    color: #2a4060;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #0f1e30;
}

/* Spinner override */
.stSpinner > div { border-top-color: #00b4ff !important; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------
# MODEL PATH
# -----------------------------------
MODEL_PATH = r"C:\Users\singh\OneDrive\Desktop\malaria_detection\malaria_detector.h5"


# -----------------------------------
# MODEL LOADER
# -----------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(50, 50, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation='sigmoid')
    ])
    model.build((None, 50, 50, 3))
    model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)
    return model


model = load_model()

class_names = ["Uninfected", "Parasitized"]
IMG_SIZE = (50, 50)


def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


# -----------------------------------
# HERO
# -----------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-badge">🔬 Deep Learning Diagnostics</div>
    <div class="hero-title">MalariaAI</div>
    <p class="hero-sub">CNN-powered blood cell analysis for malaria parasite detection</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------
# STATS
# -----------------------------------
st.markdown("""
<div class="stats-grid">
    <div class="stat-card">
        <div class="stat-value">50×50</div>
        <div class="stat-label">Input Resolution</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">CNN</div>
        <div class="stat-label">Architecture</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">2</div>
        <div class="stat-label">Output Classes</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------
# UPLOAD
# -----------------------------------
st.markdown('<div class="upload-label">📂 &nbsp;Upload Blood Cell Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# -----------------------------------
# PREDICTION FLOW
# -----------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.markdown('<div class="preview-wrap"><div class="preview-tag">SAMPLE INPUT</div>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    processed_image = preprocess_image(image)

    with st.spinner("Running inference through neural network…"):
        time.sleep(0.6)
        prediction = model.predict(processed_image, verbose=0)

    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction))
    alt_confidence = float(prediction[0][1 - predicted_index])

    is_parasitized = predicted_class == "Parasitized"
    card_class = "danger" if is_parasitized else "safe"
    icon = "⚠️" if is_parasitized else "✅"
    hint = (
        "Malaria parasites detected in this blood cell sample. Please consult a medical professional."
        if is_parasitized else
        "No parasites detected. The sample appears healthy."
    )

    # Result card
    st.markdown(f"""
    <div class="result-card {card_class}">
        <div class="result-icon">{icon}</div>
        <div class="result-status">Diagnosis Result</div>
        <div class="result-label">{predicted_class}</div>
        <div class="result-hint">{hint}</div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence bar
    conf_pct = confidence * 100
    st.markdown(f"""
    <div class="conf-wrap">
        <div class="conf-header">
            <span class="conf-title">Model Confidence</span>
            <span class="conf-value">{conf_pct:.1f}%</span>
        </div>
        <div class="conf-track">
            <div class="conf-fill {card_class}" style="width:{conf_pct:.1f}%"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Breakdown
    st.markdown(f"""
    <div class="breakdown">
        <div class="breakdown-header">Raw Prediction Scores</div>
        <div class="breakdown-row">
            <span class="breakdown-key">Uninfected probability</span>
            <span class="breakdown-val">{float(prediction[0][0])*100:.2f}%</span>
        </div>
        <div class="breakdown-row">
            <span class="breakdown-key">Parasitized probability</span>
            <span class="breakdown-val">{float(prediction[0][1])*100:.2f}%</span>
        </div>
        <div class="breakdown-row">
            <span class="breakdown-key">Predicted class</span>
            <span class="breakdown-val">{predicted_class}</span>
        </div>
        <div class="breakdown-row">
            <span class="breakdown-key">Image resolution (original)</span>
            <span class="breakdown-val">{image.size[0]} × {image.size[1]} px</span>
        </div>
        <div class="breakdown-row">
            <span class="breakdown-key">Preprocessed to</span>
            <span class="breakdown-val">50 × 50 px · RGB</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

 