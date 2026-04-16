"""streamlit_app.py WEB PAGE ALL IN ONE"""

import pandas as pd
import streamlit as st
import torch
import tensorflow as tf
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
from pathlib import Path

# Add src on path Python
sys.path.insert(0, str(Path(__file__).parent))
from src.config import NUM_CLASSES, CLASS_NAMES, NORMALIZE_MEAN, NORMALIZE_STD
from src.model_pytorch import create_model_pytorch
from src.model_tensorflow import create_model_tensorflow

# Configuration page
st.set_page_config(
    page_title="COMPUTER VISION :  Image Classifier",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

# CSS 
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; }
    .success-box { background-color: #d4edda; padding: 10px; border-radius: 8px; margin: 10px 0; }
    .big-font { font-size: 24px !important; font-weight: bold; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 20px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    with st.spinner("Loading models ... "):
        # PyTorch
        pt_model = create_model_pytorch(NUM_CLASSES)
        pt_path = Path("models/Christian_model.pth") 
        pt_model.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
        pt_model.eval()

        # TensorFlow
        tf_path = Path("models/Christian_model.keras")
        tf_model = tf.keras.models.load_model(tf_path)
    return pt_model, tf_model

# Préprocessing
def preprocess_pytorch(img):
    t = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])
    return t(img).unsqueeze(0)

def preprocess_tensorflow(img):
    img = img.resize((150, 150)).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


st.title("Project1 : Image Classifier")
st.caption("Classifying images using PyTorch & TensorFlow • streamlit • Christian")

with st.sidebar:
    st.header("⚙️ Parameters")
    model_choice = st.radio("Choose the model", ["PyTorch", "TensorFlow"], horizontal=True)
    st.divider()
    st.info("The models are loaded once  ... ")

uploaded_file = st.file_uploader("📷 Drag and drop an image here", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Imag uploaded", use_container_width=True)
    
    with col2:
        if st.button("🔍 Classify this image", type="primary", use_container_width=True):
            with st.spinner("Analysis in progress ..."):
                pt_model, tf_model = load_models()
                
                if model_choice == "PyTorch":
                    inp = preprocess_pytorch(image)
                    with torch.no_grad():
                        out = pt_model(inp)
                        probs = torch.softmax(out, dim=1)[0].numpy()
                else:
                    inp = preprocess_tensorflow(image)
                    probs = tf_model.predict(inp, verbose=0)[0]

                pred_idx = int(np.argmax(probs))
                conf = float(probs[pred_idx])
                
                st.success(f"✅Prediction : **{CLASS_NAMES[pred_idx].title()}**")
                st.metric(label="Accuracy", value=f"{conf*100:.2f}%")
                
                st.subheader("Probality distrbution")
                prob_df = pd.DataFrame({"Probabilité": probs}, index=CLASS_NAMES)
                st.bar_chart(prob_df)
else:
    st.info("👆 Upload an image to start the classification.")

#st.divider()
st.caption("Built with PyTorch & TensorFlow • Deployed on Streamlit Cloud")