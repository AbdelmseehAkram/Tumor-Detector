import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import gdown
import os
from PIL import Image
import tempfile

# ==========================================================
# 🧠 Streamlit Setup
# ==========================================================
st.set_page_config(page_title="Brain Tumor Detector", page_icon="🧠", layout="wide")
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image, and the model will predict whether it has a tumor.")

# ==========================================================
# 📥 Download and Load Model
# ==========================================================
@st.cache_resource
def download_and_load_model():
    model_url = "https://drive.google.com/uc?id=1iepaskt-97Hr9hDBoiMZmNO0-qynvnBg"
    model_path = os.path.join(tempfile.gettempdir(), "tumor_model.keras")

    if not os.path.exists(model_path):
        with st.spinner("Downloading model from Google Drive..."):
            gdown.download(model_url, model_path, quiet=False)

    try:
        model = tf.keras.models.load_model(model_path, compile=False)

        # نحاول نجيب آخر طبقة Conv تلقائيًا (من أي موديل)
        conv_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        if not conv_layers:
            st.error("❌ No Conv2D layers found — Grad-CAM won't work.")
            grad_model = None
        else:
            last_conv_layer = conv_layers[-1]
            grad_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=[last_conv_layer.output, model.output]
            )

        return model, grad_model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        raise e


model, grad_model = download_and_load_model()
st.success("✅ Model loaded successfully!")

# ==========================================================
# 🔍 Prediction + Grad-CAM
# ==========================================================
def predict_and_visualize(img: Image.Image):
    img_array = np.array(img)
    input_img = cv2.resize(img_array, (224, 224))
    input_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.expand_dims(input_img, axis=0)
    )

    preds = model.predict(input_array)
    pred_idx = np.argmax(preds[0])
    pred_label = "Tumor" if pred_idx == 0 else "Normal"
    confidence = preds[0][pred_idx] * 100

    # 🔥 Grad-CAM
    if grad_model:
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_array)
            loss = predictions[:, pred_idx]

        grads = tape.gradient(loss, conv_outputs)[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)

        # Normalize heatmap
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1

        # Resize heatmap to match image
        heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)
    else:
        overlay = img_array

    return pred_label, confidence, overlay


# ==========================================================
# 📤 Upload Section
# ==========================================================
uploaded_file = st.file_uploader("📁 Upload an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(np.array(img), caption="🧩 Uploaded Image", use_container_width=True)

    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing..."):
            label, conf, cam = predict_and_visualize(img)

        st.markdown(f"### 🧾 Prediction: **{label} ({conf:.2f}%)**")
        st.image(
            cv2.cvtColor(cam, cv2.COLOR_BGR2RGB),
            caption="🔥 Grad-CAM Visualization",
            use_container_width=True
        )
else:
    st.info("Please upload an image to begin.")

# ==========================================================
# 🧾 Footer
# ==========================================================
st.markdown("---")
st.caption("Developed by Seha | Powered by TensorFlow & Streamlit 🚀")
