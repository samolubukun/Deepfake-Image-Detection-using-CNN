import os
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# Cache the model loading to improve performance
@st.cache_resource
def load_deepfake_image_model():
    return load_model("deepfake_image_detection.h5")

# Function to predict whether the image is real or fake
def predict_deepfake_image(image_path, model):
    # Load and preprocess the image
    img = keras_image.load_img(image_path, target_size=(256, 256))
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    return 'Real' if prediction[0] > 0.5 else 'Fake'

# Streamlit app
def main():
    st.title("Deepfake Image Detection")
    st.write("Upload an image to detect if it is a real or deepfake image.")

    # File uploader for user input
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        # Save uploaded image to a temporary file
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Load the model
        model = load_deepfake_image_model()
        
        # Predict whether the image is real or fake
        result = predict_deepfake_image(temp_image_path, model)
        
        # Display the result
        if result == 'Real':
            st.success("Prediction: Real")
        else:
            st.error("Prediction: Fake")
        
        # Clean up temporary file
        os.remove(temp_image_path)

if __name__ == "__main__":
    main()
