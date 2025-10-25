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
        # 🔧 NEW APPROACH: Keep base_model separate for Grad-CAM access
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet', 
            include_top=False, 
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False
        
        # Build classification model
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu', name='dense_128')(x)
        x = tf.keras.layers.Dropout(0.3, name='dropout_03')(x)
        outputs = tf.keras.layers.Dense(2, activation='softmax', name='output')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.load_weights(weights_path)
        
        st.success("✅ Model weights loaded!")
        
        # Find last conv layer in base_model - try multiple approaches
        last_conv_layer_name = None
        
        # Approach 1: Search by type
        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                st.info(f"🎯 Found Conv2D layer: **{layer.name}**")
                break
        
        # Approach 2: Try known MobileNetV2 layers
        if not last_conv_layer_name:
            known_names = ['Conv_1', 'out_relu', 'block_16_project']
            for name in known_names:
                try:
                    layer = base_model.get_layer(name)
                    last_conv_layer_name = name
                    st.success(f"✅ Using known layer: **{name}**")
                    break
                except:
                    pass
        
        # Return both the model and base_model for Grad-CAM
        return model, base_model, last_conv_layer_name
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        import traceback
        st.code(traceback.format_exc())
        raise e


model, base_model, last_conv_layer_name = download_and_load_model()

# ==========================================================
# Grad-CAM Function (using base_model directly)
# ==========================================================
def make_gradcam_heatmap(img_array, model, base_model, last_conv_layer_name, pred_index):
    """
    Generate Grad-CAM heatmap by accessing base_model directly
    """
    if not last_conv_layer_name:
        return None
    
    try:
        # Create a model that maps input to conv layer output
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[
                base_model.get_layer(last_conv_layer_name).output,
                model.output
            ]
        )
        
        # Compute gradient of top predicted class
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, pred_index]
        
        # Extract gradients
        grads = tape.gradient(loss, conv_outputs)
        
        if grads is None:
            st.error("❌ Gradients are None!")
            return None
        
        # Compute weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the conv outputs
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        
        return heatmap.numpy()
        
    except Exception as e:
        st.error(f"❌ Grad-CAM error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


# ==========================================================
# Prediction and Visualization
# ==========================================================
def predict_and_visualize(img: Image.Image):
    img_array = np.array(img)
    
    # Preprocess
    input_img = cv2.resize(img_array, (224, 224))
    input_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.expand_dims(input_img, axis=0)
    )
    
    # Predict
    preds = model.predict(input_array, verbose=0)
    pred_idx = np.argmax(preds[0])
    
    st.write("---")
    st.write("### 📊 Prediction Probabilities:")
    st.write(f"**Class 0**: {preds[0][0]*100:.2f}%")
    st.write(f"**Class 1**: {preds[0][1]*100:.2f}%")
    st.write(f"**Predicted Index**: {pred_idx}")
    
    # Determine label (you may need to swap these based on your training)
    pred_label = "Tumor" if pred_idx == 1 else "Normal"
    confidence = preds[0][pred_idx] * 100
    
    # Generate Grad-CAM
    overlay = img_array
    
    if last_conv_layer_name:
        st.write("### 🔥 Generating Grad-CAM...")
        
        heatmap = make_gradcam_heatmap(
            input_array, 
            model, 
            base_model, 
            last_conv_layer_name, 
            pred_idx
        )
        
        if heatmap is not None:
            # Resize heatmap to match image
            heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            
            # Apply colormap
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Overlay
            overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
            st.success("✅ Grad-CAM generated successfully!")
        else:
            st.warning("⚠️ Could not generate Grad-CAM")
    else:
        st.warning("⚠️ No conv layer found - Grad-CAM disabled")
    
    return pred_label, confidence, overlay


# ==========================================================
# Upload Section
# ==========================================================
st.write("---")
uploaded_file = st.file_uploader("📁 Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(np.array(img), caption="🧩 Original Image", use_container_width=True)

    if st.button("🔍 Analyze Image", type="primary"):
        with col2:
            with st.spinner("🧠 Analyzing..."):
                label, conf, cam = predict_and_visualize(img)
            
            st.image(
                cv2.cvtColor(cam, cv2.COLOR_BGR2RGB),
                caption="🔥 Grad-CAM Visualization",
                use_container_width=True
            )
        
        st.markdown(f"## 🧾 Result: **{label}** ({conf:.2f}% confidence)")
        
        if label == "Tumor":
            st.error("⚠️ **Tumor detected!** Red areas indicate regions of concern.")
        else:
            st.success("✅ **No tumor detected.** Scan appears normal.")
            
else:
    st.info("👆 Please upload an MRI image to begin analysis.")

st.markdown("---")
st.caption("Developed by Seha | Powered by TensorFlow & Streamlit 🚀")
st.caption("⚠️ **Disclaimer**: This is for educational purposes only. Always consult medical professionals.")
