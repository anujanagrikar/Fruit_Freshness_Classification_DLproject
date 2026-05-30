import streamlit as st
from PIL import Image
from main import predict_image

# Page Settings
st.set_page_config(
    page_title="Fruit Freshness Classifier",
    page_icon="🍎",
    layout="centered"
)

# Title
st.title("🍎 Fruit Freshness Classifier")
st.write("Drag and drop a fruit image to classify it.")

# Drag & Drop Upload
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display Uploaded Image
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Predict Button
    if st.button("Predict"):

        predicted_class, confidence = predict_image(image)

        st.success(f"Prediction: {predicted_class}")
        st.info(f"Confidence: {confidence:.2f}%")