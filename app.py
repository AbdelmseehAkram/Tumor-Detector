import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import tempfile
import matplotlib.pyplot as plt
from PIL import Image
import gdown
import os

# ============================
# 🧠 Load the trained model from Google Drive
# ============================
@st.cache_resource
def load_model():
    model_path = "brain_tumor_model.h5"

    # ✅ Google Drive file ID (من الرابط بتاعك)
    file_id = "13WLOGZrL909JZePLVtpc5ZKVVfR_kZlh"
    download_url = f"https://drive.google.com/uc?id={file_id}"

    # ✅ Download model if not found locally
    if not os.path.exists(model_path):
        with st.spinner("📥 Downloading model from Google Drive..."):
            gdown.download(download_url, model_path, quiet=False)
    
    model = tf.keras.models.load_model(model_path)
    return model


# ============================
# ⚙️ Grad-CAM function
# ============================
def make_gradcam_heatmap(img_array, model):
    # Find base MobileNetV2 inside the model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is None:
        raise ValueError("❌ Could not find MobileNetV2 base model!")

    last_conv_layer = base_model.get_layer("Conv_1")

    # Reconnect model manually
    x = base_model.output
    for layer in model.layers[1:]:
        x = layer(x)
    manual_output = x

    grad_model = tf.keras.models.Model(
        [base_model.input],
        [last_conv_layer.output, manual_output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)

    # Normalize
    heatmap = np.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap)
    return heatmap.numpy(), predictions, int(pred_index)


# ============================
# 🌐 Streamlit UI
# ============================
st.set_page_config(page_title="Brain Tumor Detector", page_icon="🧠", layout="centered")
st.title("🧠 Brain Tumor Detection using MobileNetV2 + Grad-CAM")
st.write("Upload an MRI brain scan image and the model will predict and highlight possible tumor regions.")

model = load_model()

uploaded_file = st.file_uploader("📤 Upload MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # Load and preprocess image
    img = Image.open(tfile.name).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    # Generate Grad-CAM
    with st.spinner("🧩 Analyzing image..."):
        heatmap, predictions, pred_index = make_gradcam_heatmap(img_array, model)
        prob = float(predictions[0][pred_index]) * 100
        label = "Cancer" if pred_index == 1 else "Normal"

        # Overlay heatmap
        img_cv = np.array(img)
        heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

        # Add label text
        color = (0, 255, 0) if label == "Normal" else (0, 0, 255)
        cv2.putText(superimposed_img,
                    f"Diagnosis: {label} ({prob:.2f}%)",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA)

    # Show results
    st.subheader("🩸 Results:")
    st.image(img, caption="Original Image", use_container_width=True)
    st.image(superimposed_img, caption="Grad-CAM Visualization", channels="BGR", use_container_width=True)

    st.success(f"✅ **Diagnosis:** {label} ({prob:.2f}%)")
