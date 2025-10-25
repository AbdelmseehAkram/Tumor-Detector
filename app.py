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
        # 🧠 Rebuild the model architecture exactly as in training
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet', include_top=False, input_shape=(224, 224, 3)
        )
        base_model.trainable = False

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(2, activation='softmax')
        ])

        # Load weights instead of full model
        model.load_weights(weights_path)

        # 🔧 FIX: Get the correct last conv layer for MobileNetV2
        # Print all layer names to debug
        st.write("🔍 **Debug: Available layers in base_model:**")
        layer_names = [layer.name for layer in base_model.layers if 'conv' in layer.name.lower()]
        st.write(layer_names[-5:])  # Show last 5 conv layers
        
        # Try different possible last conv layer names
        conv_model = None
        head_model = None
        try:
            # MobileNetV2's actual last conv layer names
            possible_names = ["out_relu", "Conv_1", "Conv_1_bn", "block_16_project"]
            last_conv_layer = None
            
            for name in possible_names:
                try:
                    last_conv_layer = base_model.get_layer(name)
                    st.success(f"✅ Found layer: {name}")
                    break
                except:
                    continue
            
            if last_conv_layer is None:
                # Fallback: get last layer with output
                for layer in reversed(base_model.layers):
                    if len(layer.output_shape) == 4:  # Conv layer has 4D output
                        last_conv_layer = layer
                        st.success(f"✅ Using fallback layer: {layer.name}")
                        break
            
            if last_conv_layer:
                conv_model = tf.keras.models.Model(inputs=model.input, outputs=last_conv_layer.output)
                head_model = tf.keras.models.Model(inputs=last_conv_layer.output, outputs=model.output)
            
        except Exception as e:
            st.warning(f"⚠️ Grad-CAM setup warning: {e}")

        return model, conv_model, head_model
    except Exception as e:
        st.error(f"❌ Error loading weights: {e}")
        raise e


model, conv_model, head_model = download_and_load_model()
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

    preds = model.predict(input_array)
    pred_idx = np.argmax(preds[0])
    
    # 🔧 FIX: Show both class probabilities for debugging
    st.write(f"**Debug - Class 0 (Tumor?): {preds[0][0]*100:.2f}%**")
    st.write(f"**Debug - Class 1 (Normal?): {preds[0][1]*100:.2f}%**")
    
    pred_label = "Tumor" if pred_idx == 0 else "Normal"
    confidence = preds[0][pred_idx] * 100

    # Grad-CAM - Only for TUMOR predictions
    overlay = img_array
    
    if conv_model and head_model:
        try:
            input_tensor = tf.convert_to_tensor(input_array)
            
            with tf.GradientTape() as tape:
                conv_outputs = conv_model(input_tensor)
                tape.watch(conv_outputs)
                predictions = head_model(conv_outputs)
                # 🔧 FIX: Always use index 0 (Tumor class) for heatmap
                loss = predictions[:, 0]  # Changed from pred_idx to 0

            grads = tape.gradient(loss, conv_outputs)
            
            if grads is not None:
                grads = grads[0]
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
                conv_outputs_val = conv_outputs[0]
                
                # Weight the channels by the gradients
                for i in range(pooled_grads.shape[-1]):
                    conv_outputs_val[:, :, i] *= pooled_grads[i]
                
                heatmap = tf.reduce_mean(conv_outputs_val, axis=-1)
                heatmap = np.maximum(heatmap, 0)
                
                if np.max(heatmap) != 0:
                    heatmap /= np.max(heatmap)
                
                # Resize and apply colormap
                heatmap = cv2.resize(heatmap.numpy(), (img_array.shape[1], img_array.shape[0]))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                
                # Blend with original image
                overlay = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)
                st.success("✅ Grad-CAM generated successfully!")
            else:
                st.warning("⚠️ Gradients are None")
                
        except Exception as e:
            st.error(f"❌ Error in Grad-CAM: {e}")
            import traceback
            st.code(traceback.format_exc())
            overlay = img_array

    return pred_label, confidence, overlay


# ==========================================================
# Upload Section
# ==========================================================
uploaded_file = st.file_uploader("📁 Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(np.array(img), caption="🧩 Uploaded Image", use_container_width=True)

    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing..."):
            label, conf, cam = predict_and_visualize(img)

        st.markdown(f"### 🧾 Prediction: **{label} ({conf:.2f}%)**")
        st.image(
            cv2.cvtColor(cam, cv2.COLOR_BGR2RGB),
            caption="🔥 Grad-CAM Visualization (Tumor Focus)",
            use_container_width=True
        )
else:
    st.info("Please upload an image to start.")

st.markdown("---")
st.caption("Developed by Seha | Powered by TensorFlow & Streamlit 🚀")
