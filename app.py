import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import gdown
import os
from PIL import Image
import tempfile
import matplotlib.pyplot as plt

# ==========================================================
# 🧠 1. App Configuration
# ==========================================================
st.set_page_config(page_title="Brain Tumor Detector", page_icon="🧠", layout="wide")
st.title("🧠 Brain Tumor Detection using MobileNetV2")
st.write("Upload an MRI image and the model will predict whether it contains a tumor or not.")

# ==========================================================
# 📦 2. Download and Load Model
# ==========================================================
@st.cache_resource
def download_and_load_model():
    model_url = "https://drive.google.com/uc?id=1iepaskt-97Hr9hDBoiMZmNO0-qynvnBg"
    model_path = os.path.join(tempfile.gettempdir(), "tumor_model.keras")

    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(model_url, model_path, quiet=False)

    # Load the model safely
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        base_model = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                break

        if base_model is None:
            raise ValueError("❌ Base model (MobileNetV2) not found inside the model.")

        last_conv_layer = base_model.get_layer("Conv_1")

        # Build grad model for heatmap
        grad_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[last_conv_layer.output, model(base_model.output)]
        )

        return model, base_model, grad_model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        raise e


model, base_model, grad_model = download_and_load_model()
st.success("✅ Model loaded successfully!")

# ==========================================================
# 🧬 3. Prediction Function
# ==========================================================
def predict_and_visualize(img: Image.Image):
    img_array = np.array(img)
    input_img = cv2.resize(img_array, (224, 224))
    input_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.expand_dims(input_img, axis=0)
    )

    # Prediction
    predictions = model.predict(input_array)
    pred_index = np.argmax(predictions[0])
    pred_label = "Tumor" if pred_index == 0 else "Normal"
    probability = float(predictions[0][pred_index]) * 100

    # Grad-CAM visualization
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(input_array)
        loss = preds[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)

    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

    # Overlay heatmap
    img_orig = np.array(img)
    heatmap = cv2.resize(heatmap, (img_orig.shape[1], img_orig.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_orig, 0.6, heatmap, 0.4, 0)

    return pred_label, probability, superimposed_img


# ==========================================================
# 📤 4. Upload Section
# ==========================================================
uploaded_file = st.file_uploader("📁 Upload an MRI image (jpg/png/jpeg)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(np.array(img), caption="🧩 Uploaded Image", use_container_width=True)

    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing image..."):
            pred_label, prob, cam_img = predict_and_visualize(img)

        st.markdown(f"### 🧾 Prediction: **{pred_label}** ({prob:.2f}%)")

        st.image(
            cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB),
            caption="🔥 Grad-CAM Tumor Visualization",
            use_container_width=True
        )
else:
    st.info("Please upload an MRI image to begin the analysis.")

# ==========================================================
# 🧾 5. Footer
# ==========================================================
st.markdown("---")
st.caption("Developed by Seha | Powered by TensorFlow & Streamlit 🚀")
