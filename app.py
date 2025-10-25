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
        
        # 🔧 CRITICAL: We need to find the layer OUTPUT in the model graph, not base_model
        # The layers inside base_model when called in Functional API create NEW outputs
        
        # Strategy: Find conv layer by searching through model.layers
        last_conv_layer = None
        last_conv_layer_name = None
        
        st.write("🔍 Searching model layers...")
        
        # Search through ALL layers in the full model (not just base_model)
        for layer in model.layers:
            st.write(f"  - {layer.name} ({type(layer).__name__})")
            
            # If this is the base_model layer itself
            if 'mobilenetv2' in layer.name.lower() or type(layer).__name__ == 'Functional':
                st.info(f"📦 Found base model wrapper: {layer.name}")
                
                # Now search inside this layer
                for inner_layer in reversed(layer.layers):
                    if isinstance(inner_layer, tf.keras.layers.Conv2D):
                        last_conv_layer = inner_layer
                        last_conv_layer_name = inner_layer.name
                        st.success(f"✅ Found Conv2D: {inner_layer.name}")
                        break
                
                if last_conv_layer:
                    break
        
        # 🔧 Build grad_model using the layer from INSIDE the model's graph
        grad_model = None
        if last_conv_layer:
            try:
                # We need to get the output of this layer from the model's computational graph
                # Find the layer by navigating through base_model's call
                
                # Get base_model layer from model
                base_in_model = None
                for layer in model.layers:
                    if 'mobilenetv2' in layer.name.lower() or type(layer).__name__ == 'Functional':
                        base_in_model = layer
                        break
                
                if base_in_model:
                    # Get the conv layer output from base_in_model
                    conv_output = base_in_model.get_layer(last_conv_layer_name).output
                    
                    grad_model = tf.keras.Model(
                        inputs=model.inputs,
                        outputs=[conv_output, model.output]
                    )
                    
                    # Test it
                    test_input = np.random.random((1, 224, 224, 3)).astype('float32')
                    test_conv, test_pred = grad_model(test_input)
                    st.success(f"✅ Grad-CAM model created! Conv shape: {test_conv.shape}")
                else:
                    st.error("❌ Could not find base_model in model layers")
                
            except Exception as e:
                st.error(f"❌ Grad-CAM model creation failed: {e}")
                st.write("🔄 Trying alternative approach...")
                
                # Alternative: Build grad model using model internals
                try:
                    # Find the actual tensor in the graph
                    # When we call base_model(inputs), it creates intermediate tensors
                    # We need to access those through the model structure
                    
                    # Rebuild with explicit intermediate outputs
                    inputs = model.inputs[0]
                    
                    # Get base_model layer from model
                    for layer in model.layers:
                        if 'mobilenetv2' in layer.name.lower() or type(layer).__name__ == 'Functional':
                            # Call base_model to get its output
                            base_output = layer.output
                            
                            # Create a model that gives us internal layer output
                            temp_model = tf.keras.Model(
                                inputs=layer.input,
                                outputs=layer.get_layer(last_conv_layer_name).output
                            )
                            
                            # Now trace from model input through this
                            conv_output = temp_model(inputs)
                            
                            grad_model = tf.keras.Model(
                                inputs=model.inputs,
                                outputs=[conv_output, model.output]
                            )
                            
                            # Test
                            test_input = np.random.random((1, 224, 224, 3)).astype('float32')
                            test_conv, test_pred = grad_model(test_input)
                            st.success(f"✅ Grad-CAM created (alt)! Conv shape: {test_conv.shape}")
                            break
                            
                except Exception as e2:
                    st.error(f"❌ Alternative also failed: {e2}")
                    grad_model = None
        
        return model, grad_model, last_conv_layer_name
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        import traceback
        st.code(traceback.format_exc())
        raise e


model, grad_model, last_conv_layer_name = download_and_load_model()

# ==========================================================
# Grad-CAM Function (using pre-built grad_model)
# ==========================================================
def make_gradcam_heatmap(img_array, grad_model, pred_index):
    """
    Generate Grad-CAM heatmap using pre-built grad_model
    """
    if grad_model is None:
        return None
    
    try:
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
    
    if grad_model:
        st.write("### 🔥 Generating Grad-CAM...")
        
        heatmap = make_gradcam_heatmap(
            input_array, 
            grad_model,
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
        st.warning("⚠️ Grad-CAM model not available")
    
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
