import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# --- App Title and Description ---
st.set_page_config(page_title="Flower Classifier", page_icon="🌸")
st.title("🌸 Flower Classification CNN")
st.write(
    "Upload an image of a flower, and this app will predict its type "
    "using a trained Convolutional Neural Network."
)

# --- Load the Trained Model ---
# Use st.cache_resource to load the model only once and cache it.
@st.cache_resource
def load_keras_model():
    """Load the pre-trained Keras model."""
    try:
        model = load_model('my_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

model = load_keras_model()

# --- Define Class Labels ---
# These should match the classes the model was trained on.
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# --- Image Upload and Prediction ---
uploaded_file = st.file_uploader(
    "Choose a flower image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model is not None:
    # --- Preprocess the Image ---
    try:
        # Open and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("") # Add a little space

        # Resize to the model's expected input size (128x128)
        image = image.resize((128, 128))
        
        # Convert the image to a numpy array and rescale
        image_array = np.array(image) / 255.0
        
        # Add a batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # --- Make a Prediction ---
        st.write("Classifying...")
        
        prediction = model.predict(image_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        confidence = np.max(prediction)

        # --- Display the Result ---
        st.success(f"**Prediction:** {predicted_class_name.title()}")
        st.info(f"**Confidence:** {confidence:.2%}")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

# --- Add a sidebar with some information ---
st.sidebar.header("About")
st.sidebar.info(
    "This web app uses a Convolutional Neural Network (CNN) "
    "trained on a dataset of five types of flowers: daisies, "
    "dandelions, roses, sunflowers, and tulips."
)
st.sidebar.header("How to Use")
st.sidebar.markdown(
    "1. **Upload an image** of a flower using the file uploader.\n"
    "2. The app will **automatically classify** the flower.\n"
    "3. The **prediction and confidence** will be displayed."
)
