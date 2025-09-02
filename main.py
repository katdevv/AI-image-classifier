import tensorflow as tf

import os
# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 

# OpenCV for resizing images.
import cv2

# Array operations (turn PIL image → NumPy array)
import numpy as np

# builds the web app UI
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import(
    MobileNetV2, # a pre-trained CNN model
    preprocess_input, # function to scale/normalize images before feeding them into the model
    decode_predictions # converts model outputs into human-readable labels
)

# to open and handle uploaded images
from PIL import Image

def load_model():
    # Loads the MobileNetV2 model with pretrained ImageNet weights
    model = MobileNetV2(weights="imagenet")
    return model

def preprocess_image(image):
    img = np.array(image) # Convert PIL Image -> NumPy array
    img = cv2.resize(img, (224, 224)) # Resize to MobileNetV2 input size
    img = preprocess_input(img) # Normalize to match training setup
    img = np.expand_dims(img, axis=0) # Add batch dimension 
    return img

def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        # run model
        predictions = model.predict(processed_image)
        # Top 3 predictions + %s
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None

def main():
    # Configures Streamlit page
    st.set_page_config(page_title="AI Image Classifier", page_icon="🖼️", layout="centered")
    tf.get_logger().setLevel("ERROR")

    st.title("AI Image Classifier")
    st.write("Upload an image and let AI tell you what is in it!")

    # MobileNetV2 is heavy. Cache it, so Streamlit only loads once
    @st.cache_resource
    def load_cached_model():
        return load_model()

    model = load_cached_model()

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])

    if uploaded_file is not None:
        imageSt = st.image(
            uploaded_file, caption="Uploaded Image", use_container_width=True
        )
        btn = st.button("Classify Image")

        if btn:
            with st.spinner("Analyzing Image..."):
                image = Image.open(uploaded_file)
                predictions = classify_image(model, image)

                if predictions:
                    st.subheader("Predictions")
                    for _, label, score in predictions:
                        st.write(f"**{label}**: {score:.2%}")

if __name__ == "__main__":
    main()