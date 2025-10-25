import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import gdown
import os
from PIL import Image

# ==============================================================
# ⚙️ 1️⃣ Page Configuration
# ==============================================================
st.set_page_config(page_title="🧠 Brain Tumor Detector", layout="centered")
st.title("🧠 Brain Tumor Detection using MobileNetV2 + Grad-CAM")

# ==============================================================
# 🧩 2️⃣ Model Download and Load
# ==============================================================
@st.cache_resource
def download_and_load_model():
    model_path = "tumor_detector.keras"

    # Download from Google Drive if not found
    if not os.path.exists(model_path):
        st.info("📥 Downloading model from Google Drive ...")
        url = "https://drive.google.com/uc?id=1iepaskt-97Hr9hDBoiMZmNO0-qynvnBg"  # ✅ direct link
        gdown.download(url, model_path, quiet=False)

    # Load the model
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        raise e

    # Try building model (important for Sequential)
    try:
        _ = model(tf.zeros((1, 224, 224, 3)))
    except Exception:
        st.warning("⚠️ Model not pre-built, skipping input build.")

    # Detect base model (MobileNetV2)
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is None:
        st.warning("⚠️ No base model found — using full model.")
        base_model = model

    # Find last conv layer
    last_conv_layer = None
    for layer in base_model.layers[::-1]:
        if len(layer.output_shape) == 4:
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError("❌ No convolutional layer found!")

    st.write(f"✅ Found last conv layer: `{last_conv_layer.name}`")

    # Build Grad-CAM model safely
    try:
        grad_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[last_conv_layer.output, model.output]
        )
    except Exception:
        st.warning("⚠️ Using fallback Grad-CAM path.")
        grad_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[last_conv_layer.output, base_model.output]
        )

    return model, base_model, grad_model


# ==============================================================
# 🧠 3️⃣ Grad-CAM Function
# ==============================================================
def generate_gradcam(grad_model, img_array, orig_img, class_names):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)

    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # Overlay heatmap
    heatmap = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)

    label = class_names[pred_index]
    prob = float(predictions[0][pred_index]) * 100

    cv2.putText(superimposed_img, f"{label}: {prob:.2f}%",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

    return superimposed_img, label, prob


# ==============================================================
# 🚀 4️⃣ App Logic
# ==============================================================
st.divider()
st.header("📸 Upload MRI Image")

uploaded_file = st.file_uploader("Upload an MRI image of the brain (jpg/png)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🧩 Uploaded Image", use_container_width=True)

    # Prepare image
    img_array = np.array(img)
    input_img = cv2.resize(img_array, (224, 224))
    input_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.expand_dims(input_img, axis=0)
    )

    # Load model
    model, base_model, grad_model = download_and_load_model()

    # Predict
    st.info("🧠 Running prediction and generating Grad-CAM...")
    class_names = ["Tumor", "Normal"]

    try:
        result_img, label, prob = generate_gradcam(grad_model, input_array, img_array, class_names)
        st.image(result_img, caption=f"Prediction: {label} ({prob:.2f}%)", use_container_width=True)
        st.success(f"✅ Model Prediction: **{label} ({prob:.2f}%)**")
    except Exception as e:
        st.error(f"⚠️ Error during Grad-CAM generation: {e}")

else:
    st.warning("📤 Please upload an MRI image to continue.")
