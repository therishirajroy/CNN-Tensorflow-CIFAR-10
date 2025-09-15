import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from streamlit_option_menu import option_menu


st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="centered"
)

@st.cache_resource
def load_keras_model():
    """Loads the pre-trained Keras model."""
    try:
        model = tf.keras.models.load_model('cifar10nn.keras')
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Load the model
model = load_keras_model()

# Define the CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# Sidebar Navigation
with st.sidebar:
    selection = option_menu("Main Menu", ["Model","Inference","About Me"], icons = ["gear","star",'person'], menu_icon = "menu", default_index = 0,
                               orientation = "vertical")

# About Page
if selection == "Model":
    st.title("CIFAR-10 Object Classification Using CNN")
    st.markdown("""
    This application is a web-based interface for a **Convolutional Neural Network (CNN)** trained on the famous [**CIFAR-10 dataset**](https://www.cs.toronto.edu/~kriz/cifar.html).

    ### What is CIFAR-10?
    The CIFAR-10 dataset is a collection of 60,000 small (32x32 pixels) color images, categorized into 10 distinct classes:
    - ✈️ Airplane
    - 🚗 Automobile
    - 🐦 Bird
    - 🐱 Cat
    - 🦌 Deer
    - 🐶 Dog
    - 🐸 Frog
    - 🐴 Horse
    - 🚢 Ship
    - 🚚 Truck

    ### Model Architecture
    The deep learning model used here is a standard CNN. CNNs are specifically designed to process pixel data and are highly effective for image recognition tasks. 
    The model building and evaluation is in this [link](https://colab.research.google.com/drive/16GEkZlAWuwTHLu-OgImslHMcqYCDQW6t?usp=sharing).
    The architecture typically includes:
    1.  **Convolutional Layers:** To automatically and adaptively learn spatial hierarchies of features from the input images.
    2.  **Pooling Layers:** To reduce the dimensionality of each feature map but retain the most important information.
    3.  **Fully Connected Layers:** To perform classification based on the high-level features extracted by the convolutional and pooling layers.
    
    """)
    image = Image.open("CNN.png")

    st.image(image, caption="My CNN Architecture", width=500)

    st.markdown("""
                The model was built using TensorFlow/Keras and saved as `cifar10nn.keras`. You can interact with it on the **Inference** page!
                ### How to Use This App
                1. Navigate to the **Inference** page using the sidebar.
                2. Upload an image of an object that you want to classify.
                3. The model will predict the class of the object and display the result along with the confidence level.
                """)



# Inference Page
elif selection == "Inference":
    st.title("Model Inference 🚀")
    st.header("Image Classifier")

    if model is not None:
        # Upload file
        uploaded_file = st.file_uploader("Upload an image of an object...", type=["jpg", "jpeg", "png", "avif"])

        if uploaded_file is not None:
            # Open and display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_container_width=True)
            st.write("")

            with st.spinner('Classifying...'):
                img_resized = image.resize((32, 32))
                img_array = np.array(img_resized)
                if img_array.shape[2] == 4: # Handle RGBA images
                    img_array = img_array[:, :, :3]
                img_array = img_array / 255.0
                img_batch = np.expand_dims(img_array, axis=0)

                # Make a prediction
                prediction = model.predict(img_batch)
                predicted_class_index = np.argmax(prediction, axis=1)[0]
                predicted_class_name = CLASS_NAMES[predicted_class_index]
                confidence = np.max(prediction) * 100

                # Display the prediction
                st.success(f"**Predicted Label:** {predicted_class_name.capitalize()}")
                st.info(f"**Confidence:** {confidence:.2f}%")
    else:
        st.warning("The model file 'cifar10nn.keras' could not be loaded. Please make sure it's in the correct directory.")


# Student Info Page
elif selection == "About Me":
    st.title("About Me 🧑‍💻")
    st.markdown("""
    ---
    - **Name:** Rishiraj Roy
    - **Roll No:** 2409008
    - **Course:** M.Sc. Big Data Analytics
    - **Semester:** III
    - **College:** St. Xavier's College, Mumbai
    ---
    """)