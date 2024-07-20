import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms

st.title("🔎 Concrete Crack Detector")

# Load model
FILE = 'model.pth'
model = torch.load(FILE)
model.eval()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button('Detect'):
        # Predict
        input_image = transform(image).unsqueeze(0)
        with torch.no_grad():
            z = model(input_image)
            _, yhat = torch.max(z.data, 1)

        # Display the prediction result
        st.write(f"Prediction: {'Crack Detected' if yhat.item() == 1 else 'No Crack Detected'}")
