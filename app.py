import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import gdown
import os
from PIL import Image
import tempfile

# ==========================================================
# Streamlit Config
# ==========================================================
st.set_page_config(page_title="Brain Tumor Detector", page_icon="🧠", layout="wide")
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image to predict whether it has a tumor.")

# ==========================================================
# Download and Load Model from Google Drive
# ==========================================================
@st.cache_resource
def download_and_load_model():
    model_url = "https://drive.google.com/uc?id=1iepaskt-97Hr9hDBoiMZmNO0-qynvnBg"
    weights_path = os.path.join(tempfile.gettempdir(), "tumor_weights.keras")

    # Download from Google Drive
    if not os.path.exists(weights_path):
        with st.spinner("📥 Downloading model weights from Google Drive..."):
            gdown.download(model_url, weights_path, quiet=False)

    try:
        # 🧠 Rebuild the model using Functional API for Grad-CAM
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet', include_top=False, input_shape=(224, 224, 3)
        )
        base_model.trainable = False

        # 🔧 FIX: Use Functional API instead of Sequential
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Load weights
        model.load_weights(weights_path)

        # 🔧 FIX: Get the last conv layer from base_model properly
        # Find the last convolutional layer in MobileNetV2
        last_conv_layer = None
        for layer in reversed(base_model.layers):
            # Check if layer has output_shape attribute first
            if hasattr(layer, 'output_shape') and layer.output_shape is not None:
                if len(layer.output_shape) == 4:  # Convolutional layer
                    last_conv_layer = layer
                    st.info(f"🎯 Using layer for Grad-CAM: **{layer.name}**")
                    break
        
        # Create Grad-CAM model
        grad_model = None
        if last_conv_layer:
            grad_model = tf.keras.Model(
                inputs=[model.inputs],
                outputs=[last_conv_layer.output, model.output]
            )

        return model, grad_model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        import traceback
        st.code(traceback.format_exc())
        raise e


model, grad_model = download_and_load_model()
st.success("✅ Model loaded successfully!")

# ==========================================================
# Prediction and Grad-CAM Visualization
# ==========================================================
def predict_and_visualize(img: Image.Image):
    img_array = np.array(img)
    input_img = cv2.resize(img_array, (224, 224))
    input_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.expand_dims(input_img, axis=0)
    )

    # Get predictions
    preds = model.predict(input_array, verbose=0)
    pred_idx = np.argmax(preds[0])
    
    # 🔧 Show both class probabilities for debugging
    st.write(f"**📊 Class 0 probability: {preds[0][0]*100:.2f}%**")
    st.write(f"**📊 Class 1 probability: {preds[0][1]*100:.2f}%**")
    st.write(f"**🎯 Predicted class index: {pred_idx}**")
    
    pred_label = "Tumor" if pred_idx == 0 else "Normal"
    confidence = preds[0][pred_idx] * 100

    # Generate Grad-CAM heatmap
    overlay = img_array
    
    if grad_model is not None:
        try:
            input_tensor = tf.convert_to_tensor(input_array)
            
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(input_tensor)
                # 🔧 Focus on Tumor class (index 0) OR the predicted class
                # Change this to 0 if you always want tumor heatmap, or pred_idx for predicted class
                class_channel = predictions[:, pred_idx]  # Use pred_idx or change to 0 for tumor

            # Compute gradients
            grads = tape.gradient(class_channel, conv_outputs)
            
            if grads is not None:
                # Global average pooling on gradients
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                
                # Weight the conv outputs by the gradients
                conv_outputs = conv_outputs[0]
                heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap)
                
                # Normalize heatmap
                heatmap = tf.maximum(heatmap, 0)
                if tf.reduce_max(heatmap) != 0:
                    heatmap = heatmap / tf.reduce_max(heatmap)
                
                heatmap = heatmap.numpy()
                
                # Resize to original image size
                heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
                heatmap = np.uint8(255 * heatmap)
                
                # Apply colormap
                heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                
                # Blend with original image
                overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
                
                st.success("✅ Grad-CAM heatmap generated!")
            else:
                st.warning("⚠️ Could not compute gradients")
                
        except Exception as e:
            st.error(f"❌ Grad-CAM error: {e}")
            import traceback
            st.code(traceback.format_exc())

    return pred_label, confidence, overlay


# ==========================================================
# Upload Section
# ==========================================================
uploaded_file = st.file_uploader("📁 Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(np.array(img), caption="🧩 Original Image", use_container_width=True)

    if st.button("🔍 Analyze Image", type="primary"):
        with st.spinner("🧠 Analyzing brain MRI..."):
            label, conf, cam = predict_and_visualize(img)

        st.markdown(f"### 🧾 Prediction: **{label}** (Confidence: **{conf:.2f}%**)")
        
        with col2:
            st.image(
                cv2.cvtColor(cam, cv2.COLOR_BGR2RGB),
                caption="🔥 Grad-CAM Heatmap",
                use_container_width=True
            )
        
        # Interpretation
        if label == "Tumor":
            st.error("⚠️ **Tumor detected!** The highlighted regions show areas of concern.")
        else:
            st.success("✅ **No tumor detected.** The brain scan appears normal.")
            
else:
    st.info("👆 Please upload an MRI image to start the analysis.")

st.markdown("---")
st.caption("Developed by Seha | Powered by TensorFlow & Streamlit 🚀")
