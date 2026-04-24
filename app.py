import streamlit as st
import numpy as np
from PIL import Image
from utils import load_colorization_models, preprocess_and_colorize
import os

# Suppress TensorFlow logging warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI Image Colorization",
    page_icon="🎨",
    layout="centered"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-family: 'Inter', sans-serif;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        color: #888;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    .branding {
        text-align: center;
        margin-top: 60px;
        padding-top: 20px;
        border-top: 1px solid #333;
        color: #888;
        font-size: 0.95rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    .stButton > button {
        border-radius: 8px;
        transition: all 0.3s ease;
        font-weight: bold;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(78, 205, 196, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("<h1 class='main-title'>Historical Image Colorization</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Bring your black & white photos to life using Deep Learning</p>", unsafe_allow_html=True)

# --- LOAD MODELS ---
with st.spinner("Loading AI Models... This might take a moment on first run."):
    color_model, inception = load_colorization_models()

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Upload a grayscale image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load the uploaded image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Display the uploaded image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("<h4 style='text-align: center;'>Original Upload</h4>", unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Colorize button
        if st.button("Colorize Image 🪄", type="primary", use_container_width=True):
            with st.spinner("Analyzing and colorizing image..."):
                try:
                    # Run inference pipeline
                    gray_img, color_img = preprocess_and_colorize(img_array, color_model, inception)
                    
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown("<h3 style='text-align: center;'>Results</h3>", unsafe_allow_html=True)
                    
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        st.markdown("**Grayscale Input (Resized)**")
                        st.image(gray_img, use_container_width=True, clamp=True)
                        
                    with res_col2:
                        st.markdown("**Colorized Output**")
                        st.image(color_img, use_container_width=True, clamp=True)
                        
                    st.success("✨ Colorization complete!")
                except Exception as e:
                    st.error(f"An error occurred during colorization: {e}")
                    
    except Exception as e:
        st.error(f"Error loading the image: {e}")

# --- BRANDING ---
st.markdown("<div class='branding'>Made by Akshat Karode</div>", unsafe_allow_html=True)