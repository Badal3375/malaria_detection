# ============================================
# FIX KeyError: 'select_mode'
# Problem:
# Your JSON language files do not contain all required keys
# Solution:
# Use default fallback dictionary so app NEVER crashes
# ============================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import json
import os

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Malaria AI",
    page_icon="🧬",
    layout="centered"
)

# ============================================
# DEFAULT LANGUAGE KEYS (SAFE FALLBACK)
# ============================================
DEFAULT_LANG = {
    "title": "MalariaAI Detection System",
    "select_mode": "Select Analysis Mode",
    "blood_mode": "Blood Cell Analysis",
    "dna_mode": "DNA Testing",
    "blood_subtitle": "Blood Cell Analysis Mode",
    "dna_subtitle": "DNA Testing Mode",
    "upload_blood": "Upload Blood Cell Image",
    "upload_dna": "Upload DNA Test Image (PCR/Gel)",
    "uploaded_sample": "Uploaded Sample",
    "running_ai": "Running AI Diagnosis...",
    "result": "Result",
    "dna_positive": "Malaria DNA detected. Please consult a doctor.",
    "dna_negative": "No malaria DNA detected.",
    "blood_positive": "Malaria parasites detected in blood smear.",
    "blood_negative": "No malaria parasites detected.",
    "confidence": "Model Confidence",
    "breakdown": "Prediction Breakdown",
    "predicted_class": "Predicted Class",
    "analysis_mode": "Analysis Mode",
    "original_resolution": "Original Resolution",
    "preprocessed_resolution": "Preprocessed Resolution",
    "footer": "MalariaAI • Deep Learning Powered Detection System"
}

# ============================================
# LANGUAGE LOADER
# ============================================
def load_language(lang_code):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    lang_path = os.path.join(BASE_DIR, "languages", f"{lang_code}.json")

    # Start with defaults
    lang_data = DEFAULT_LANG.copy()

    # Load custom language file if exists
    if os.path.exists(lang_path):
        try:
            with open(lang_path, "r", encoding="utf-8") as file:
                user_lang = json.load(file)

            # Merge custom keys with defaults
            lang_data.update(user_lang)

        except Exception as e:
            st.warning(f"Language file error: {e}")

    else:
        st.warning(f"{lang_code}.json not found. Using default English.")

    return lang_data

# ============================================
# LANGUAGE OPTIONS
# ============================================
languages = {
    "English": "en",
    "Hindi": "hi",
    "Punjabi": "pa",
    "Haryanvi": "ha"
    
}

selected_language = st.sidebar.selectbox(
    "Select Language",
    list(languages.keys())
)

lang = load_language(languages[selected_language])

# ============================================
# MODE SELECTOR
# ============================================
mode = st.sidebar.selectbox(
    lang.get("select_mode", DEFAULT_LANG["select_mode"]),
    [
        lang.get("blood_mode", DEFAULT_LANG["blood_mode"]),
        lang.get("dna_mode", DEFAULT_LANG["dna_mode"])
    ]
)

is_dna_mode = mode == lang.get("dna_mode", DEFAULT_LANG["dna_mode"])

# ============================================
# MODEL PATH
# ============================================
MODEL_PATH = r"C:\Users\singh\OneDrive\Desktop\malaria_detection\malaria_detector.h5"

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(50, 50, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.build((None, 50, 50, 3))
    model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)

    return model

model = load_model()

# ============================================
# CLASS NAMES
# ============================================
class_names = (
    ["Negative", "Positive"]
    if is_dna_mode else
    ["Uninfected", "Parasitized"]
)

IMG_SIZE = (50, 50)

# ============================================
# IMAGE PREPROCESS
# ============================================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image, dtype=np.float32) / 255.0

    return np.expand_dims(img_array, axis=0)

# ============================================
# UI
# ============================================
st.title(lang.get("title"))

st.subheader(
    lang.get("dna_subtitle")
    if is_dna_mode
    else lang.get("blood_subtitle")
)

# ============================================
# FILE UPLOADER
# ============================================
uploaded_file = st.file_uploader(
    lang.get("upload_dna")
    if is_dna_mode
    else lang.get("upload_blood"),

    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)

# ============================================
# PREDICTION
# ============================================
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption=lang.get("uploaded_sample"),
        use_container_width=True
    )

    processed_image = preprocess_image(image)

    with st.spinner(lang.get("running_ai")):
        time.sleep(1)
        prediction = model.predict(processed_image, verbose=0)

    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]

    confidence = float(np.max(prediction))
    conf_pct = confidence * 100

    # RESULTS
    if is_dna_mode:
        if predicted_class == "Positive":
            st.error(f"🧬 {lang.get('result')}: {predicted_class}")
            st.warning(lang.get("dna_positive"))
        else:
            st.success(f"✅ {lang.get('result')}: {predicted_class}")
            st.info(lang.get("dna_negative"))

    else:
        if predicted_class == "Parasitized":
            st.error(f"⚠️ {lang.get('result')}: {predicted_class}")
            st.warning(lang.get("blood_positive"))
        else:
            st.success(f"✅ {lang.get('result')}: {predicted_class}")
            st.info(lang.get("blood_negative"))

    # CONFIDENCE
    st.progress(confidence)

    st.write(
        f"### {lang.get('confidence')}: {conf_pct:.2f}%"
    )

    # BREAKDOWN
    st.write(f"## {lang.get('breakdown')}")

    st.write(
        f"**{class_names[0]} Probability:** "
        f"{float(prediction[0][0]) * 100:.2f}%"
    )

    st.write(
        f"**{class_names[1]} Probability:** "
        f"{float(prediction[0][1]) * 100:.2f}%"
    )

    st.write(
        f"**{lang.get('predicted_class')}:** {predicted_class}"
    )

    st.write(
        f"**{lang.get('analysis_mode')}:** {mode}"
    )

    st.write(
        f"**{lang.get('original_resolution')}:** "
        f"{image.size[0]} × {image.size[1]} px"
    )

    st.write(
        f"**{lang.get('preprocessed_resolution')}:** "
        f"{IMG_SIZE[0]} × {IMG_SIZE[1]} px"
    )

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption(lang.get("footer"))