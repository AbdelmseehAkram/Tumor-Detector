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
        # 🔧 COMPLETE REWRITE: Build everything to support Grad-CAM properly
        
        # Step 1: Load base model
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet', include_top=False, input_shape=(224, 224, 3)
        )
        base_model.trainable = False
        
        # Step 2: Find the last convolutional layer NAME in base model
        last_conv_layer_name = None
        for layer in reversed(base_model.layers):
            if 'Conv' in type(layer).__name__:
                last_conv_layer_name = layer.name
                st.info(f"🎯 Found conv layer: **{layer.name}** (type: {type(layer).__name__})")
                break
        
        if not last_conv_layer_name:
            # Try known layer names as fallback
            for name in ['out_relu', 'Conv_1', 'block_16_project']:
                try:
                    base_model.get_layer(name)
                    last_conv_layer_name = name
                    st.success(f"✅ Using known layer: **{name}**")
                    break
                except:
                    continue
        
        # Step 3: Build the full model using Functional API
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Step 4: Load weights
        model.load_weights(weights_path)
        st.success("✅ Model weights loaded!")
        
        # Step 5: Build Grad-CAM model if we found a conv layer
        grad_model = None
        if last_conv_layer_name:
            try:
                # Create a model that outputs both conv layer and final prediction
                last_conv_layer = base_model.get_layer(last_conv_layer_name)
                
                # Build a new model that maps inputs to conv output and predictions
                grad_model = tf.keras.Model(
                    inputs=model.inputs,
                    outputs=[base_model.get_layer(last_conv_layer_name).output, model.output]
                )
                
                # Test the model
                test_input = tf.random.normal((1, 224, 224, 3))
                test_conv, test_pred = grad_model(test_input)
                st.success(f"✅ Grad-CAM model ready! Conv shape: {test_conv.shape}")
                
            except Exception as e:
                st.error(f"❌ Grad-CAM setup failed: {e}")
                grad_model = None

        return model, grad_model, last_conv_layer_name
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        import traceback
        st.code(traceback.format_exc())
        raise e


model, grad_model, last_conv_layer_name = download_and_load_model()
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
    
    st.write("---")
    st.write("### 🔬 Grad-CAM Debug Info:")
    
    if grad_model is None:
        st.error("❌ Grad-CAM model is None! Cannot generate heatmap.")
        return pred_label, confidence, overlay
    
    st.info(f"✅ Grad-CAM model exists")
    
    try:
        input_tensor = tf.convert_to_tensor(input_array)
        st.info(f"✅ Input tensor shape: {input_tensor.shape}")
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_tensor)
            st.info(f"✅ Conv outputs shape: {conv_outputs.shape}")
            st.info(f"✅ Predictions shape: {predictions.shape}")
            
            # 🔧 Focus on predicted class
            class_channel = predictions[:, pred_idx]
            st.info(f"✅ Class channel value: {class_channel.numpy()[0]:.4f}")

        # Compute gradients
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            st.error("❌ Gradients are None! This means the gradient computation failed.")
            st.warning("💡 Possible reasons: layer not trainable, or disconnected from output")
            return pred_label, confidence, overlay
        
        st.success(f"✅ Gradients computed! Shape: {grads.shape}")
        st.info(f"📊 Gradient stats: min={tf.reduce_min(grads):.6f}, max={tf.reduce_max(grads):.6f}, mean={tf.reduce_mean(grads):.6f}")
        
        # Global average pooling on gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        st.info(f"✅ Pooled gradients shape: {pooled_grads.shape}")
        
        # Weight the conv outputs by the gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        st.info(f"✅ Heatmap shape before normalization: {heatmap.shape}")
        st.info(f"📊 Heatmap stats: min={tf.reduce_min(heatmap):.6f}, max={tf.reduce_max(heatmap):.6f}")
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0)
        heatmap_max = tf.reduce_max(heatmap)
        
        if heatmap_max == 0:
            st.error("❌ Heatmap is all zeros! Cannot normalize.")
            return pred_label, confidence, overlay
        
        heatmap = heatmap / heatmap_max
        heatmap = heatmap.numpy()
        
        st.success(f"✅ Heatmap normalized! Values range: {heatmap.min():.3f} to {heatmap.max():.3f}")
        
        # Resize to original image size
        heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        
        st.info(f"✅ Heatmap resized to: {heatmap.shape}")
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Blend with original image
        overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
        
        st.success("🎉 Grad-CAM heatmap generated successfully!")
            
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
