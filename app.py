import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '2' filters out INFO and WARNING messages

# Create a list of class names based on your dataset folders
class_names = sorted(os.listdir("Dataset/images.cv_jzk6llhf18tm3k0kyttxz/data/test"))

# Function to load the best model
@st.cache_resource
def load_best_model():
    # Load the MobileNet model that you saved
    model_path = 'models/mobilenet_finetuned.h5'
    model = load_model(model_path)
    return model

# Load the model
model = load_best_model()

st.title('🐟 Multiclass Fish Image Classifier')
st.write('Upload an image of a fish, and the model will predict its species.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image for the model
    img_array = tf.image.resize(np.array(image), (224, 224))
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0  # Rescale to [0, 1]

    # Make a prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: **{confidence:.2f}%**")

    st.write("---")
    st.write("### All Predictions")
    # Display all class probabilities
    prediction_df = pd.DataFrame({
        'Species': class_names,
        'Confidence': [f'{p * 100:.2f}%' for p in score.numpy()]
    }).sort_values('Confidence', ascending=False)
    st.dataframe(prediction_df)