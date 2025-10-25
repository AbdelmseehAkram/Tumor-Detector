import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import gdown
import os

# 🔹 Load model from Google Drive
@st.cache_resource
def load_model():
    model_path = "tumor_model.keras"
    if not os.path.exists(model_path):
        file_id = "1MNzIKsB2VKNleMR2hQzBMpeD69eiF0ej"  # Google Drive file ID
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        raise e

    return model


# Load model
model = load_model()

# Get model input shape automatically
input_shape = model.input_shape[1:3]  # e.g. (128, 128)
st.write(f"🧩 Model expects input size: {input_shape}")

# 🎨 Streamlit UI
st.title("🧠 Brain Tumor Detection App")
st.write("Upload an MRI image to check if a brain tumor is detected.")

uploaded_file = st.file_uploader("📤 Upload an Image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        image_path = tmp.name

    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, input_shape)  # ✅ dynamically uses correct size
    image_array = np.expand_dims(image_resized / 255.0, axis=0)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 🔍 Prediction
    prediction = model.predict(image_array)
    result = "✅ No Tumor Detected" if prediction[0][0] < 0.5 else "⚠️ Tumor Detected"

    st.subheader("Prediction Result:")
    st.write(result)
