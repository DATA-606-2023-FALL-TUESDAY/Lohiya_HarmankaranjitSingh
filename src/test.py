# Streamlit Web App
import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import os 	
from PIL import Image

# Print current working directory for debugging
print("Current Working Directory:", os.getcwd())

# Load the trained model 
model_path = r"C:\Users\Karan's\Desktop\Project files\CNN_Model.h5"
model = tf.keras.models.load_model(model_path) 

# Function to predict the class of the X-ray image
def predict_image(img):
    img_array = np.array(img)
    
    # Check if the image is grayscale
    if len(img_array.shape) == 2:
        img_resized = cv2.resize(img_array, (128, 128))
        img_resized = np.expand_dims(img_resized, axis=-1)  # Add channel dimension
    elif len(img_array.shape) == 3:
        img_resized = cv2.resize(img_array, (128, 128))
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        img_resized = np.expand_dims(img_resized, axis=-1)  # Add channel dimension
    else:
        # Handle unexpected number of channels
        st.warning("Unexpected number of channels in the input image.")
        return None

    img_reshaped = img_resized.reshape(-1, 128, 128, 1)
    prediction = model.predict(img_reshaped)

    if prediction > 0.5:
        return "Pneumonia"
    else:
        return "Normal"


# Main Streamlit App
def main():
    st.title("Pneumonia Detection from X-ray Images")
    st.write("This tool is designed to assist in diagnosing Pneumonia based on X-ray images. However, always consult with a healthcare professional for medical advice.")
    
    # Upload Image
    img_file = st.file_uploader("Upload an X-ray Image", type=['jpg', 'png', 'jpeg'])
    
    # If an image is uploaded
    if img_file:
        # img = Image.open(img_file)
        img = Image.open(os.path.join(os.getcwd(), img_file.name))
        st.image(img, caption='Uploaded X-ray Image', use_column_width=True)
        
        # Predict Button
        if st.button("Predict"):
            result = predict_image(img)
            
            # Display result
            if result == "Pneumonia":
                st.warning(f"Model Prediction: {result}")
                st.write("The X-ray image shows signs of Pneumonia. Please consult with a healthcare professional.")
            else:
                st.success(f"Model Prediction: {result}")
                st.write("The X-ray image appears to be Normal. However, for any health concerns, always consult with a healthcare professional.")
                
    st.write("## About")
    st.write("This application uses a Convolutional Neural Network (CNN) model trained on a dataset of X-ray images to predict the presence of Pneumonia. It serves as a supplementary tool and should not replace professional medical advice.")

# Run the app
if __name__ == '__main__':
    main()
