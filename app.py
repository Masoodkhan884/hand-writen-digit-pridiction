import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Load the model
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Streamlit App
st.set_page_config(page_title="Handwritten Digit Classifier", layout="centered")

st.title("🖊️ Handwritten Digit Detection")
st.markdown("Upload a **28x28** grayscale image of a digit (0–9) for prediction.")

# Upload Image
uploaded_file = st.file_uploader("Choose a digit image (PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L').resize((28, 28))
    
    # Show image
    st.image(image, caption="Uploaded Image", use_column_width=False, width=150)

    # Preprocess
    img_array = np.array(image)
    img_array = 1 - img_array / 255.0  # Invert and normalize
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.success(f"✅ **Predicted Digit:** `{predicted_class}`")
    
    # Show prediction probabilities
    st.subheader("🔢 Prediction Probabilities")
    st.bar_chart(prediction[0])
