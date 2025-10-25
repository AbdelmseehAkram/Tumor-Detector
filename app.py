# ==========================================================
# Tumor Detection Streamlit App with Grad-CAM
# ==========================================================
import os
import tempfile
import gdown
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image as keras_image

# ==========================================================
# App Config
# ==========================================================
st.set_page_config(
    page_title="Tumor Detection with Grad-CAM",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🧠 Brain Tumor Detection with Grad-CAM")
st.markdown("Upload an MRI/CT image and get prediction with heatmap visualization.")

# ==========================================================
# Load Model from Google Drive (weights only)
# ==========================================================
@st.cache_resource
def download_and_load_model():
    # Google Drive file ID
    model_url = "https://drive.google.com/uc?id=1iepaskt-97Hr9hDBoiMZmNO0-qynvnBg"
    weights_path = os.path.join(tempfile.gettempdir(), "tumor_weights.keras")

    # Download weights if not exist
    if not os.path.exists(weights_path):
        with st.spinner("📥 Downloading model weights from Google Drive..."):
            gdown.download(model_url, weights_path, quiet=False)

    try:
        # Build base MobileNetV2
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
        base_model.trainable = False

        # Add classification head
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        output = tf.keras.layers.Dense(2, activation='softmax')(x)

        model = tf.keras.Model(inputs=base_model.input, outputs=output)
        model.load_weights(weights_path)

        # Grad-CAM model
        last_conv_layer = base_model.get_layer("Conv_1")
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [last_conv_layer.output, model.output]
        )

        return model, grad_model
    except Exception as e:
        st.error(f"❌ Error loading weights: {e}")
        raise e

model, grad_model = download_and_load_model()
st.success("✅ Model loaded successfully!")

# ==========================================================
# Image Upload
# ==========================================================
uploaded_file = st.file_uploader("Upload MRI/CT Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="🧩 Uploaded Image", use_column_width=True)

    # ======================================================
    # Preprocess Image
    # ======================================================
    img_resized = cv2.resize(img_rgb, (224,224))
    img_array = keras_image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # ======================================================
    # Predict
    # ======================================================
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    classes = ["Cancer", "Normal"]
    pred_label = classes[class_idx]
    probability = float(preds[0][class_idx]*100)
    st.write(f"**Prediction:** {pred_label} ({probability:.2f}%)")

    # ======================================================
    # Grad-CAM Functions
    # ======================================================
    def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
        with tf.GradientTape() as tape:
            last_conv_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_output = last_conv_output[0]

        heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def overlay_heatmap(heatmap, image, alpha=0.4):
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0)
        return overlay

    # ======================================================
    # Generate Grad-CAM Heatmap
    # ======================================================
    heatmap = make_gradcam_heatmap(img_array, grad_model, class_idx)
    overlay_img = overlay_heatmap(heatmap, img_rgb)
    st.image(overlay_img, caption="🧠 Grad-CAM Heatmap", use_column_width=True)
