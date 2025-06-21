import streamlit as st
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from generate import generate_digit_images

st.set_page_config(page_title="Digit Generator")
st.title("🧠 Handwritten Digit Image Generator")
st.caption("Generate MNIST-like digit images using your trained model.")

digit = st.selectbox("Choose a digit to generate (0-9):", list(range(10)))

if st.button("Generate Images"):
    images = generate_digit_images(digit, num_images=5)

    fig, ax = plt.subplots()
    grid = make_grid(images, nrow=5, normalize=True)
    ax.imshow(grid.permute(1, 2, 0).cpu().numpy())
    ax.axis("off")
    st.pyplot(fig)
