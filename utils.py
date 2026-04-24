import numpy as np
from skimage.transform import resize
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
import streamlit as st

@st.cache_resource
def load_colorization_models():
    """
    Loads the Keras colorization model and the InceptionResNetV2 embedding model.
    Cached via Streamlit so they are only loaded once.
    """
    color_model = load_model('best_colorizer.h5', compile=False)
    inception = InceptionResNetV2(weights='imagenet', include_top=True)
    return color_model, inception

def preprocess_and_colorize(img_array, color_model, inception):
    """
    Preprocesses the input image, extracts features using InceptionResNetV2,
    and runs the prediction using the colorization model.
    
    Returns:
    - grayscaled_display: 256x256 grayscale image for display
    - colorized_img: 256x256 RGB image with predicted colors
    """
    # 1. Normalize values
    if img_array.max() > 1.0:
        img_array = img_array.astype('float32') / 255.0
    else:
        img_array = img_array.astype('float32')
        
    # 2. Resize to 256x256 (required by the colorization model)
    img_resized = resize(img_array, (256, 256), mode='reflect', anti_aliasing=True)
    
    # 3. Ensure image has 3 channels for uniform processing
    if len(img_resized.shape) == 2:
        img_resized = gray2rgb(img_resized)
    elif img_resized.shape[2] == 1:
        img_resized = gray2rgb(img_resized[:, :, 0])
    elif img_resized.shape[2] == 4:
        img_resized = img_resized[:, :, :3]
        
    # 4. Force purely grayscale 3-channel image 
    # (Matches what the model was trained on: grayscaled RGB inputs)
    grayscaled = rgb2gray(img_resized)
    grayscaled_rgb = gray2rgb(grayscaled)
    
    # 5. Extract InceptionResNetV2 embedding
    embed_input = resize(grayscaled_rgb, (299, 299, 3), mode='constant')
    embed_input = np.expand_dims(embed_input, axis=0)
    embed_input = preprocess_input(embed_input * 255.0) # preprocess_input expects 0-255 pixels
    embed = inception.predict(embed_input)
    
    # 6. Convert to LAB color space
    lab_img = rgb2lab(grayscaled_rgb)
    L_channel = lab_img[:, :, 0]
    L_channel_input = L_channel.reshape((1, 256, 256, 1))
    
    # 7. Predict the AB color channels
    output = color_model.predict([L_channel_input, embed])
    output = output * 128
    
    # 8. Reconstruct the image combining L and predicted AB
    reconstructed_lab = np.zeros((256, 256, 3))
    reconstructed_lab[:, :, 0] = L_channel_input[0, :, :, 0]
    reconstructed_lab[:, :, 1:] = output[0]
    
    # 9. Convert back to RGB for display
    colorized_img = lab2rgb(reconstructed_lab)
    
    # Clip values to ensure valid displayable RGB [0, 1]
    colorized_img = np.clip(colorized_img, 0.0, 1.0)
    grayscaled_display = np.clip(grayscaled_rgb, 0.0, 1.0)
    
    return grayscaled_display, colorized_img
