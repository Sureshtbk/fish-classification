import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set page configuration
st.set_page_config(page_title="Fish Classifier", page_icon="🐟", layout="wide")

# Use cache to load the model only once
@st.cache_resource
def load_model(model_path):
    """Loads the trained Keras model."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocesses the uploaded image to be model-ready."""
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    img_array = img_array / 255.0 # Rescale
    return img_array

# Define class names (ensure the order matches the training class indices)
CLASS_NAMES = [
    'animal fish', 
    'animal fish bass', 
    'fish_sea_food_black_sea_sprat', 
    'fish_sea_food_gilt_head_bream', 
    'fish_sea_food_hourse_mackerel', 
    'fish_sea_food_red_mullet', 
    'fish_sea_food_red_sea_bream', 
    'fish_sea_food_sea_bass', 
    'fish_sea_food_shrimp', 
    'fish_sea_food_striped_red_mullet', 
    'fish_sea_food_trout'
] # Replace with your actual class names

# --- App Layout ---
st.title("🐟 Fish Image Classification")
st.markdown("Upload an image of a fish, and the model will predict its species.")

# Load the best model
MODEL_PATH = 'MobileNetV2_fish_classifier.h5' # Replace with your best model's path
model = load_model(MODEL_PATH)

if model:
    # File uploader
    uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)
            st.write("")

        with col2:
            st.write("### Prediction")
            with st.spinner('Classifying...'):
                # Preprocess and predict
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)
                
                # Get predicted class and confidence
                predicted_class_index = np.argmax(prediction)
                predicted_class_name = CLASS_NAMES[predicted_class_index]
                confidence = np.max(prediction) * 100
                
                st.success(f"**Predicted Species:** {predicted_class_name}")
                st.info(f"**Confidence:** {confidence:.2f}%")
                
                # Display confidence scores for all classes
                st.write("### Confidence Scores")
                st.bar_chart(prediction[0])
else:
    st.warning("Model could not be loaded. Please check the model path and file integrity.")

