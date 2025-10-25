import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import os
import gdown
from PIL import Image

# =============================
# 📦 تحميل الموديل من Google Drive
# =============================
@st.cache_resource
def load_model():
    model_path = "brain_tumor_model.h5"
    if not os.path.exists(model_path):
        file_id = "1MNzIKsB2VKNleMR2hQzBMpeD69eiF0ej"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
    model = tf.keras.models.load_model(model_path)
    return model


# =============================
# 🧠 تجهيز الصورة
# =============================
def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img, (224, 224))  # نفس حجم تدريب الموديل
    img_array = np.expand_dims(img_resized / 255.0, axis=0)
    return img_array, img


# =============================
# 🔍 رسم الـ Bounding Box
# =============================
def highlight_tumor_area(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = image.copy()
    for c in contours:
        if cv2.contourArea(c) > 200:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return result


# =============================
# 🎨 واجهة Streamlit
# =============================
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image and the model will detect if there is a tumor.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

model = load_model()

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array, original_img = preprocess_image(image)
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.error(f"⚠️ Tumor Detected! Confidence: {prediction * 100:.2f}%")
        highlighted = highlight_tumor_area(original_img)
        st.image(cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB),
                 caption="Detected Tumor Area", use_container_width=True)
    else:
        st.success(f"✅ No Tumor Detected. Confidence: {(1 - prediction) * 100:.2f}%")
