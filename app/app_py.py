# Streamlit App for Industrial Defect Detection

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from models.model_definition import load_trained_model  # Make sure this exists

# Page config
st.set_page_config(page_title="Industrial Defect Detection", page_icon="🧠")

# Class names and model path
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in-scale', 'scratches']
MODEL_PATH = os.path.join("models", "resnet18_finetuned.pth")

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_trained_model(MODEL_PATH, device)

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Streamlit interface
st.title("🧠 Industrial Surface Defect Detection")
st.markdown("Upload a metal surface image — the model will classify the defect and visualize Grad-CAM heatmap.")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, 0)
        pred_class = CLASS_NAMES[pred_idx.item()]

    st.subheader(f"🧾 Predicted Defect: **{pred_class}**  |  Confidence: {conf.item():.2f}")

    # Grad-CAM visualization
    rgb_img = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
    target_layers = [model.model.layer4[-1]]  # Adjust if needed
    cam = GradCAM(model=model.model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=img_tensor)[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    st.image(visualization, caption="🔥 Grad-CAM Defect Region", use_container_width=True)

    # Class probabilities
    st.markdown("### Class Probabilities")
    for name, prob in zip(CLASS_NAMES, probs):
        st.write(f"{name}: {prob.item():.2f}")
else:
    st.info("👆 Upload an image to start defect detection.")

st.markdown("---")
st.markdown("🧩 Developed by Sourodeep Mondal | ResNet18 Fine-tuned on NEU Surface Defect Dataset")
