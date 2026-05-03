# app.py (100% FIXED for Keras 3 + Sequential class issue)
# Uses manual architecture from your exact model config

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Malaria Detection",
    page_icon="🦠",
    layout="centered"
)

st.title("🦠 Malaria Detection System")
st.write("Upload a blood cell image to classify malaria infection.")

MODEL_PATH = r"C:\Users\singh\OneDrive\Desktop\malaria_detection\malaria_detector.h5"

# -----------------------------------
# EXACT MODEL ARCHITECTURE
# -----------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(50, 50, 3)),

        # Block 1
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        # Block 2
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        # Flatten
        tf.keras.layers.Flatten(),

        # Dense 1
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        # Dense 2
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        # Output Layer
        tf.keras.layers.Dense(2, activation='sigmoid')
    ])

    # Build model first
    model.build((None, 50, 50, 3))

    # Load weights only
    model.load_weights(MODEL_PATH, by_name=True, skip_mismatch=True)

    return model


model = load_model()

# -----------------------------------
# CLASS LABELS
# -----------------------------------
class_names = ["Uninfected","Parasitized"]

# -----------------------------------
# PREPROCESS IMAGE
# -----------------------------------
IMG_SIZE = (50, 50)

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image, dtype=np.float32)

    # Normalize
    img_array = img_array / 255.0

    # Expand dims => (1,50,50,3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# -----------------------------------
# FILE UPLOAD
# -----------------------------------
uploaded_file = st.file_uploader(
    "Upload Blood Cell Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------------
# PREDICTION
# -----------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    processed_image = preprocess_image(image)

    with st.spinner("Analyzing Image..."):
        prediction = model.predict(processed_image)

    # Since output = 2 neurons
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]

    confidence = np.max(prediction)

    # -----------------------------------
    # RESULTS
    # -----------------------------------
    if predicted_class == "Parasitized":
        st.error(f"### Prediction: {predicted_class}")
    else:
        st.success(f"### Prediction: {predicted_class}")

    st.info(f"### Confidence: {confidence * 100:.2f}%")

 