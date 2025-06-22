# app.py

import streamlit as st
import torch
import matplotlib.pyplot as plt
from generator_model import Generator

# Set up
st.set_page_config(page_title="Digit Generator", layout="centered")
st.title("🧠 MNIST Digit Generator")
st.write("Select a digit (0–9) to generate 5 fake handwritten samples.")

# Load generator
@st.cache_resource
def load_model():
    model = Generator()
    model.load_state_dict(torch.load("mnist_generator.pth", map_location="cpu"))
    model.eval()
    return model

generator = load_model()

# User input
digit = st.selectbox("Choose a digit", list(range(10)), index=0)
generate = st.button("Generate Images")

# Generate 5 images of the selected digit
def generate_images(digit, generator, n_images=5):
    z = torch.randn(n_images, 100)
    labels = torch.full((n_images,), digit, dtype=torch.long)
    with torch.no_grad():
        images = generator(z, labels).detach().cpu()
    return images

# Plot and display
def show_images(images):
    fig, axs = plt.subplots(1, len(images), figsize=(10, 2))
    for i, img in enumerate(images):
        axs[i].imshow(img.squeeze(), cmap="gray")
        axs[i].axis("off")
    st.pyplot(fig)

# Trigger generation
if generate:
    images = generate_images(digit, generator)
    show_images(images)
