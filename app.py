import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(r"C:\Users\Umaima Tariq\Dropbox\My PC (DESKTOP-CVKKAE6)\Downloads\cavity_dec_new\cavity_dec\fine_tuned_model_cavity.h5")
    return model

# Function to perform prediction
def predict(image, model):
    # Preprocess the image
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (512, 512))
    image = np.expand_dims(image, axis=0)
    
    # Perform prediction
    prediction = model.predict(image)
    return prediction[0]

# Function to display the results with boxes around cavities
def display_results(original_image, prediction, threshold):
    # Convert PIL image to numpy array
    original_image_np = np.array(original_image)

    # Resize prediction to match the shape of original image
    prediction_resized = cv2.resize(prediction, original_image_np.shape[:2][::-1])
    
    # Apply threshold
    prediction_binary = (prediction_resized > threshold).astype(np.uint8)
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(prediction_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around contours
    overlay = original_image_np.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle
    
    # Convert the overlaid image back to PIL format for display
    overlay_pil = Image.fromarray(overlay)
    
    # Display the overlaid image
    st.image(overlay_pil, caption='Original Image with Boxes around Cavities', use_column_width=True)

# Main function
def main():
    st.title('Teeth Caries Detection')

    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.sidebar.header('Adjust Threshold')
        threshold = st.sidebar.slider('Threshold', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        
        model = load_model()
        original_image = Image.open(uploaded_image)
        st.sidebar.image(original_image, caption='Uploaded Image', use_column_width=True)

        if st.button('Detect'):
            prediction = predict(original_image, model)
            display_results(original_image, prediction, threshold)

if __name__ == '__main__':
    main()
