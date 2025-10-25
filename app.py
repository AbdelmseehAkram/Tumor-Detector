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
st.write("Upload image to predict whether it has a tumor.")

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
        # 🔥 COMPLETE REBUILD: Flatten the architecture for Grad-CAM
        
        # Load MobileNetV2 and get its layers
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet', 
            include_top=False, 
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False
        
        # Find last conv layer NAME
        last_conv_layer_name = None
        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                st.info(f"🎯 Target conv layer: **{layer.name}**")
                break
        
        if not last_conv_layer_name:
            # Fallback to known names
            for name in ['Conv_1', 'out_relu']:
                try:
                    base_model.get_layer(name)
                    last_conv_layer_name = name
                    st.success(f"✅ Using fallback: **{name}**")
                    break
                except:
                    pass
        
        # 🔥 NEW APPROACH: Build model with DIRECT access to all layers
        # Instead of nesting, we'll expose the conv layer directly
        
        inputs = tf.keras.Input(shape=(224, 224, 3), name='input_image')
        
        # Pass through MobileNetV2
        x = base_model(inputs, training=False)
        
        # Get the last conv layer output directly from base_model
        last_conv_output = base_model.get_layer(last_conv_layer_name).output
        
        # Continue with classification head
        x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
        x = tf.keras.layers.Dense(128, activation='relu', name='dense_128')(x)
        x = tf.keras.layers.Dropout(0.3, name='dropout')(x)
        outputs = tf.keras.layers.Dense(2, activation='softmax', name='predictions')(x)
        
        # Main model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='tumor_classifier')
        
        # Load weights
        model.load_weights(weights_path)
        st.success("✅ Model weights loaded!")
        
        # 🔥 Build separate Grad-CAM model
        # The trick: we need ONE unified model for gradient flow
        grad_model = None
        
        if last_conv_layer_name:
            try:
                # 🎯 KEY INSIGHT: Build a SINGLE model with both outputs from the SAME graph
                # Not two separate models!
                
                # Start fresh with inputs
                grad_inputs = tf.keras.Input(shape=(224, 224, 3))
                
                # Pass through base_model (this creates the computational graph)
                grad_x = base_model(grad_inputs, training=False)
                
                # Get conv layer output from THIS graph (not from base_model directly)
                # We need to access the intermediate tensor in THIS call
                
                # The base_model when called creates internal tensors
                # We can access them through the layer's output in this specific call
                
                # Continue with the rest of the model
                grad_x2 = tf.keras.layers.GlobalAveragePooling2D()(grad_x)
                grad_x2 = tf.keras.layers.Dense(128, activation='relu')(grad_x2)
                grad_x2 = tf.keras.layers.Dropout(0.3)(grad_x2)
                grad_predictions = tf.keras.layers.Dense(2, activation='softmax')(grad_x2)
                
                # Now build a model that outputs BOTH the conv features AND predictions
                # We get conv features by creating a sub-model
                conv_layer_output = base_model.get_layer(last_conv_layer_name).output
                
                # Create intermediate model to extract conv output
                base_model_for_grad = tf.keras.Model(
                    inputs=base_model.input,
                    outputs=[conv_layer_output, base_model.output]
                )
                
                # Build the full grad model
                conv_features, _ = base_model_for_grad(grad_inputs)
                
                # Now trace through the classification head
                x_from_conv = tf.keras.layers.GlobalAveragePooling2D()(conv_features)
                x_from_conv = tf.keras.layers.Dense(128, activation='relu')(x_from_conv)
                x_from_conv = tf.keras.layers.Dropout(0.3)(x_from_conv) 
                final_predictions = tf.keras.layers.Dense(2, activation='softmax')(x_from_conv)
                
                grad_model = tf.keras.Model(
                    inputs=grad_inputs,
                    outputs=[conv_features, final_predictions]
                )
                
                # Load the SAME weights into this new model
                grad_model.load_weights(weights_path)
                
                # Verify
                test_data = np.random.random((1, 224, 224, 3)).astype('float32')
                test_conv, test_pred = grad_model(test_data)
                st.success(f"✅ Grad-CAM ready! Conv: {test_conv.shape}, Pred: {test_pred.shape}")
                
            except Exception as e:
                st.error(f"❌ Grad-CAM build failed: {e}")
                import traceback
                st.code(traceback.format_exc())
                grad_model = None
        
        return model, grad_model, last_conv_layer_name
        
    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        st.code(traceback.format_exc())
        raise e


model, grad_model, last_conv_layer_name = download_and_load_model()

# ==========================================================
# Grad-CAM Function
# ==========================================================
def make_gradcam_heatmap(img_array, grad_model, pred_index):
    """Generate Grad-CAM heatmap"""
    if grad_model is None:
        return None
    
    try:
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, pred_index]
        
        grads = tape.gradient(loss, conv_outputs)
        
        if grads is None:
            st.error("❌ Gradients are None")
            return None
        
        # Weighted combination
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_mean(conv_outputs[0] * pooled_grads, axis=-1)
        
        # Normalize
        heatmap = tf.maximum(heatmap, 0)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)
        
        return heatmap.numpy()
        
    except Exception as e:
        st.error(f"❌ Grad-CAM error: {e}")
        return None


# ==========================================================
# Prediction Function
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
    
    st.write("### 📊 Predictions:")
    st.write(f"**Class 0**: {preds[0][0]*100:.2f}%")
    st.write(f"**Class 1**: {preds[0][1]*100:.2f}%")
    
    # Label (adjust if needed: swap 0 and 1 if labels are reversed)
    pred_label = "Tumor" if pred_idx == 0 else "Normal"
    confidence = preds[0][pred_idx] * 100
    
    # Generate Grad-CAM
    overlay = img_array
    
    if grad_model:
        heatmap = make_gradcam_heatmap(input_array, grad_model, pred_idx)
        
        if heatmap is not None:
            # Resize and colorize
            heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Blend
            overlay = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
            st.success("✅ Grad-CAM generated!")
    
    return pred_label, confidence, overlay


# ==========================================================
# UI
# ==========================================================
st.write("---")
uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="🧩 Original", use_container_width=True)

    if st.button("🔍 Analyze", type="primary"):
        with st.spinner("🧠 Analyzing..."):
            label, conf, cam = predict_and_visualize(img)
        
        with col2:
            st.image(
                cv2.cvtColor(cam, cv2.COLOR_BGR2RGB),
                caption="🔥 Grad-CAM",
                use_container_width=True
            )
        
        st.markdown(f"## 🧾 Result: **{label}** ({conf:.2f}%)")
        
        if label == "Tumor":
            st.error("⚠️ **Tumor detected!** Red regions show areas of concern.")
        else:
            st.success("✅ **No tumor detected.**")
else:
    st.info("👆 Upload an image to start")

st.markdown("---")
st.caption("Developed by Abdelmseeh | TensorFlow & Streamlit 🚀")
st.caption("⚠️ Educational purposes only. Consult medical professionals.")

