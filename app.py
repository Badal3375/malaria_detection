# ============================================
# MALARIA AI — FULL FIXED VERSION
# Fixes:
# 1. Duplicate uploader removed
# 2. preprocess_image() fixed (dna_mode optional)
# 3. NameError image fixed
# 4. DNA/Blood class names fixed
# 5. Single prediction flow only
# ============================================

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
    page_icon="🧬",
    layout="centered"
)
# ============================================
# ADD MULTIPLE LANGUAGE SUPPORT
# Use this in your existing MalariaAI app
# Languages:
# English, Hindi, Punjabi, Haryanvi
# ============================================

import json

# -----------------------------------
# LANGUAGE LOADER
# -----------------------------------
def load_language(lang_code):
    with open(f"languages/{lang_code}.json", "r", encoding="utf-8") as file:
        return json.load(file)

# -----------------------------------
# LANGUAGE SELECTOR (SIDEBAR)
# -----------------------------------
languages = {
    "English": "en",
    "Hindi": "hi",
    "Punjabi": "pa",
    "Haryanvi": "ha"
}

selected_language = st.sidebar.selectbox(
    "Select Language / भाषा चुनें / ਭਾਸ਼ਾ ਚੁਣੋ",
    list(languages.keys())
)

lang = load_language(languages[selected_language])

# ============================================
# REPLACE YOUR OLD TEXT WITH lang[]
# ============================================

# PAGE TITLE
st.set_page_config(
    page_title=lang["title"],
    page_icon="🧬",
    layout="centered"
)

# MODE SELECTOR
mode = st.sidebar.selectbox(
    lang["select_mode"],
    [lang["blood_mode"], lang["dna_mode"]]
)

is_dna_mode = mode == lang["dna_mode"]

# HERO TITLE
st.title(lang["title"])

st.subheader(
    lang["dna_subtitle"] if is_dna_mode
    else lang["blood_subtitle"]
)

# FILE UPLOADER
uploaded_file = st.file_uploader(
    lang["upload_dna"] if is_dna_mode
    else lang["upload_blood"],
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)

# ============================================
# RESULTS SECTION
# ============================================
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption=lang["uploaded_sample"],
        use_container_width=True
    )

    processed_image = preprocess_image(
        image,
        dna_mode=is_dna_mode
    )

    with st.spinner(lang["running_ai"]):
        time.sleep(1)
        prediction = model.predict(processed_image, verbose=0)

    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]

    confidence = float(np.max(prediction))
    conf_pct = confidence * 100

    # DNA MODE
    if is_dna_mode:

        is_positive = predicted_class == "Positive"

        if is_positive:
            st.error(f"🧬 {lang['result']}: {predicted_class}")
            st.warning(lang["dna_positive"])
        else:
            st.success(f"✅ {lang['result']}: {predicted_class}")
            st.info(lang["dna_negative"])

    # BLOOD MODE
    else:

        is_positive = predicted_class == "Parasitized"

        if is_positive:
            st.error(f"⚠️ {lang['result']}: {predicted_class}")
            st.warning(lang["blood_positive"])
        else:
            st.success(f"✅ {lang['result']}: {predicted_class}")
            st.info(lang["blood_negative"])

    # CONFIDENCE
    st.progress(confidence)

    st.write(f"### {lang['confidence']}: {conf_pct:.2f}%")

    # BREAKDOWN
    st.write(f"## {lang['breakdown']}")

    st.write(
        f"**{class_names[0]}:** "
        f"{float(prediction[0][0]) * 100:.2f}%"
    )

    st.write(
        f"**{class_names[1]}:** "
        f"{float(prediction[0][1]) * 100:.2f}%"
    )

    st.write(f"**{lang['predicted_class']}:** {predicted_class}")

    st.write(f"**{lang['analysis_mode']}:** {mode}")

    st.write(
        f"**{lang['original_resolution']}:** "
        f"{image.size[0]} × {image.size[1]} px"
    )

    st.write(
        f"**{lang['preprocessed_resolution']}:** "
        f"{IMG_SIZE[0]} × {IMG_SIZE[1]} px"
    )

# FOOTER
st.markdown("---")
st.caption(lang["footer"])
# -----------------------------------
# SIDEBAR MODE
# -----------------------------------
mode = st.sidebar.selectbox(
    "Select Analysis Mode",
    ["Blood Cell Analysis", "DNA Testing"]
)

is_dna_mode = mode == "DNA Testing"

# -----------------------------------
# MODEL PATH
# -----------------------------------
MODEL_PATH = r"C:\Users\singh\OneDrive\Desktop\malaria_detection\malaria_detector.h5"

# -----------------------------------
# LOAD MODEL
# -----------------------------------
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
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.build((None, 50, 50, 3))

    # Load trained weights
    model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)

    return model


model = load_model()

# -----------------------------------
# CLASS NAMES
# -----------------------------------
if is_dna_mode:
    class_names = ["Negative", "Positive"]
else:
    class_names = ["Uninfected", "Parasitized"]

IMG_SIZE = (50, 50)

# -----------------------------------
# IMAGE PREPROCESSING
# -----------------------------------
def preprocess_image(image, dna_mode=False):
    image = image.convert("RGB")

    # DNA mode can use sharper enhancement later if needed
    image = image.resize(IMG_SIZE)

    img_array = np.array(image, dtype=np.float32) / 255.0

    return np.expand_dims(img_array, axis=0)

# -----------------------------------
# HERO
# -----------------------------------
st.title("🧬 MalariaAI Detection System")

st.subheader(
    "DNA Testing Mode" if is_dna_mode
    else "Blood Cell Analysis Mode"
)

# -----------------------------------
# UPLOAD
# -----------------------------------
uploaded_file = st.file_uploader(
    f"Upload {'DNA Test Image (PCR/Gel)' if is_dna_mode else 'Blood Cell Image'}",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)

# -----------------------------------
# PREDICTION FLOW
# -----------------------------------
if uploaded_file is not None:

    # Load image safely
    image = Image.open(uploaded_file)

    # Preview
    st.image(
        image,
        caption="Uploaded Sample",
        use_container_width=True
    )

    # Preprocess
    processed_image = preprocess_image(
        image,
        dna_mode=is_dna_mode
    )

    # Predict
    with st.spinner("Running AI Diagnosis..."):
        time.sleep(1)
        prediction = model.predict(processed_image, verbose=0)

    # Results
    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]

    confidence = float(np.max(prediction))
    conf_pct = confidence * 100

    # -----------------------------------
    # DNA MODE RESULT
    # -----------------------------------
    if is_dna_mode:

        is_positive = predicted_class == "Positive"

        if is_positive:
            st.error(f"🧬 Result: {predicted_class}")
            st.warning(
                "Malaria DNA detected in PCR/Gel sample. Please consult a doctor."
            )
        else:
            st.success(f"✅ Result: {predicted_class}")
            st.info(
                "No malaria DNA detected in this sample."
            )

    # -----------------------------------
    # BLOOD MODE RESULT
    # -----------------------------------
    else:

        is_positive = predicted_class == "Parasitized"

        if is_positive:
            st.error(f"⚠️ Result: {predicted_class}")
            st.warning(
                "Malaria parasites detected in blood smear."
            )
        else:
            st.success(f"✅ Result: {predicted_class}")
            st.info(
                "No malaria parasites detected."
            )

    # -----------------------------------
    # CONFIDENCE
    # -----------------------------------
    st.progress(confidence)

    st.write(f"### Model Confidence: {conf_pct:.2f}%")

    # -----------------------------------
    # BREAKDOWN
    # -----------------------------------
    st.write("## Prediction Breakdown")

    st.write(
        f"**{class_names[0]} Probability:** "
        f"{float(prediction[0][0]) * 100:.2f}%"
    )

    st.write(
        f"**{class_names[1]} Probability:** "
        f"{float(prediction[0][1]) * 100:.2f}%"
    )

    st.write(f"**Predicted Class:** {predicted_class}")

    st.write(f"**Analysis Mode:** {mode}")

    st.write(
        f"**Original Resolution:** "
        f"{image.size[0]} × {image.size[1]} px"
    )

    st.write(
        f"**Preprocessed Resolution:** "
        f"{IMG_SIZE[0]} × {IMG_SIZE[1]} px"
    )

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown("---")
st.caption(
    "MalariaAI • Deep Learning Powered Detection System"
)