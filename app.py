import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import gdown
import os
from tensorflow.keras.preprocessing import image

# ==========================================================
# ⚙️ Page Config
# ==========================================================
st.set_page_config(page_title="Brain Tumor Detector 🧠", layout="centered", page_icon="🧬")

st.title("🧠 Brain Tumor Detector with Grad-CAM")
st.write("Upload an MRI image to detect if there's a tumor and visualize the affected area.")

# ==========================================================
# 📦 Download Model from Google Drive
# ==========================================================
@st.cache_resource
def download_and_load_model():
    model_path = "tumor_detector.keras"

    # Check if model already exists locally
    if not os.path.exists(model_path):
        st.info("📥 Downloading model from Google Drive...")
        url = "https://drive.google.com/uc?id=1iepaskt-97Hr9hDBoiMZmNO0-qynvnBg"  # your link
        gdown.download(url, model_path, quiet=False)

    # Load model
    model = tf.keras.models.load_model(model_path, compile=False)
    _ = model(tf.zeros((1, 224, 224, 3)))  # build model

    # Find base model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is None:
        raise ValueError("❌ Base model (MobileNetV2) not found inside the loaded model!")

    base_index = model.layers.index(base_model)
    classifier_layers = model.layers[base_index + 1:]
    last_conv_layer = base_model.get_layer("Conv_1")

    # Build Grad-CAM model
    x = last_conv_layer.output
    for layer in classifier_layers:
        x = layer(x)
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, x]
    )

    return model, base_model, grad_model


model, base_model, grad_model = download_and_load_model()
st.success("✅ Model loaded successfully!")

# ==========================================================
# 📤 Upload Image
# ==========================================================
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    img_path = temp_file.name

    # Preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    st.image(img, caption="🩺 Uploaded MRI Image", use_column_width=True)

    # ==========================================================
    # 🔍 Grad-CAM Calculation
    # ==========================================================
    with st.spinner("Analyzing image..."):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)

        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(cv2.resize(heatmap, (224, 224)), cv2.COLORMAP_JET)

        # Overlay heatmap on original image
        img_orig = cv2.imread(img_path)
        img_orig = cv2.resize(img_orig, (224, 224))
        superimposed_img = cv2.addWeighted(img_orig, 0.6, heatmap, 0.4, 0)

        # Prediction label
        pred_label = "🧠 Tumor Detected" if pred_index == 0 else "✅ Normal Brain"
        probability = float(predictions[0][pred_index]) * 100

    # ==========================================================
    # 🖼️ Display Results
    # ==========================================================
    st.subheader("🔍 Diagnosis Result")
    st.write(f"**Prediction:** {pred_label}")
    st.write(f"**Confidence:** {probability:.2f}%")

    st.image(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB),
             caption="🔥 Grad-CAM Visualization",
             use_column_width=True)

else:
    st.info("📤 Please upload an image to start analysis.")
