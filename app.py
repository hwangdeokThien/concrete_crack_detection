import streamlit as st
from PIL import Image

st.title("🔎 Concrete Crack Detector")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button('Detect'):
        st.write("Processing the image...")

        # process the image here
        gray_image = image.convert('L')
        st.image(gray_image, caption='Processed Image (Grayscale).',
                 use_column_width=True)
