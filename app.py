import streamlit as st
from skimage import io

import numpy as np
import pandas as pd
import cv2
import pickle
from tensorflow import keras
import PIL

@st.cache_data
def load_data():
    return pd.read_csv("signname.csv").values[:, 1]

@st.cache_resource
def load_cnn_model():
    return keras.models.load_model("best_CNN_model.h5")

def translate_image(image, max_trans=5, height=32, width=32):
    translate_x = max_trans * np.random.uniform() - max_trans / 2
    translate_y = max_trans * np.random.uniform() - max_trans / 2
    translation_mat = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
    trans = cv2.warpAffine(image, translation_mat, (height, width))
    return trans


def preprocess_image(image):
    # # Load the image
    # image = io.imread(image_path)

    # Resize the image to 32x32
    image_resized = translate_image(image)

    # Convert the image to grayscale
    image_gray = np.sum(image_resized / 3, axis=2, keepdims=True)

    # Normalize the image
    image_gray_norm = (image_gray - 32) / 32

    # Expand dimensions to match the input shape of the model
    image_gray_norm = np.expand_dims(image_gray_norm, axis=0)
    
    return image_gray_norm


def predict_image(image, model):
    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Predict the class
    prediction = model.predict(preprocessed_image)

    # Get the class with the highest probability
    predicted_class = np.argmax(prediction, axis=1)

    return predicted_class

st.title("Recognition of Traffic Signs")

model = load_cnn_model()
sign_names = load_data()

uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=False)
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.")
    image = io.imread(uploaded_file)
    predicted_class = predict_image(image, model)
    st.write(f"The predicted class is: {predicted_class}")
    st.write(f"The predicted sign name is: {sign_names[predicted_class[0]]}")
