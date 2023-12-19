import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import streamlit as st
# Load the model
url = "https://huggingface.co/olzhas1997/pneumonia_model/blob/main/pneumonia_model.keras"
loaded_model = load_model(url)

# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert('L')
    img = img.resize((224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

# Streamlit app
def main():
    st.title("Pneumonia Detection Web App")

    uploaded_file = st.file_uploader("Choose an image...", type="png")

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Perform prediction
        new_image = load_and_preprocess_image(uploaded_file)
        result = loaded_model.predict(new_image)

        # Display the prediction result
        if result[0][0] > 0.5:
            prediction = 'PNEUMONIA'
        else:
            prediction = 'NORMAL'

        st.write("Predicted class:", prediction)

if __name__ == "__main__":
    main()