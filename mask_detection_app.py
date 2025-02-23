import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import io

# Load your model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("C:\\Users\\Aayushi Puri\\OneDrive\\Desktop\\projects\\Face mask detection\\new_detector\\mask_detector.model")
    return model

model = load_model()

st.title("Face Mask Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Predicting...")

    # Preprocess the image
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    # Make prediction
    prediction = model.predict(image)
    label = "Mask Detected" if prediction[0][0] > prediction[0][1] else "No Mask Detected"

    st.write(f"Prediction: {label}")