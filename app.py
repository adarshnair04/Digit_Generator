import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import make_grid
from PIL import Image
import io

# ===============================================================
# 1. CONSTANTS AND MODEL ARCHITECTURE
#    (Must be identical to your training script)
# ===============================================================

# Constants from the training script
LATENT_DIM = 100
NUM_CLASSES = 10
HIDDEN_DIM = 256
IMG_DIM = 28 * 28
CHANNELS = 1
IMG_SIZE = 28
DEVICE = torch.device("cpu")

class Generator(nn.Module):
    """
    The Generator class with the architecture matching your trained model.
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, NUM_CLASSES)
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM + NUM_CLASSES, HIDDEN_DIM),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(HIDDEN_DIM * 2, IMG_DIM),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embed = self.label_emb(labels)
        gen_input = torch.cat([noise, label_embed], dim=1)
        img = self.net(gen_input)
        img = img.view(img.size(0), CHANNELS, IMG_SIZE, IMG_SIZE)
        return img

# ===============================================================
# 2. CACHED MODEL LOADING
#    (Loads the model only once for better performance)
# ===============================================================

@st.cache_resource
def load_model():
    """
    Loads the trained generator model from disk.
    The @st.cache_resource decorator ensures this function runs only once.
    """
    try:
        model_path = 'models/cgan_generator.pth'
        generator = Generator().to(DEVICE)
        generator.load_state_dict(torch.load(model_path, map_location=DEVICE))
        generator.eval()
        return generator
    except FileNotFoundError:
        st.error(f"Model file not found at '{model_path}'. Please ensure the file is in the correct directory.")
        return None

generator = load_model()

# ===============================================================
# 3. STREAMLIT UI AND APP LOGIC
# ===============================================================

# --- Sidebar Controls ---
with st.sidebar:
    st.title("⚙️ Controls")
    st.markdown("Configure the image generation settings.")

    digit_to_generate = st.selectbox(
        label="**1. Choose a digit (0-9):**",
        options=list(range(10))
    )

    num_images = st.slider(
        label="**2. Number of images to generate:**",
        min_value=1,
        max_value=10,
        value=5  # Default value
    )

    generate_button = st.button("Generate Images", type="primary")

    with st.expander("ℹ️ How does this work?"):
        st.write("""
            This app uses a **Conditional Generative Adversarial Network (cGAN)**
            trained on the MNIST dataset. The model learned to generate
            new, synthetic images of handwritten digits based on the
            label you provide (the digit 0-9).
        """)

# --- Main Panel ---
st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using your trained model.")

# --- Image Generation Logic ---
if generate_button and generator is not None:
    st.subheader(f"Generated images of digit: {digit_to_generate}")

    with st.spinner("Generating..."):
        # Create random noise and labels for the selected digit
        noise = torch.randn(num_images, LATENT_DIM, device=DEVICE)
        labels = torch.full((num_images,), digit_to_generate, dtype=torch.long, device=DEVICE)

        # Generate images using the model
        with torch.no_grad():
            generated_imgs = generator(noise, labels)

        # Un-normalize the images from [-1, 1] back to [0, 1]
        generated_imgs = 0.5 * generated_imgs + 0.5

        # Create a grid of images using torchvision
        grid = make_grid(generated_imgs, nrow=num_images, normalize=True)
        
        # Convert the grid tensor to a PIL Image for display and download
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        pil_img = Image.fromarray((grid_np * 255).astype(np.uint8))

        # Create an in-memory byte stream to hold the image data
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        byte_im = buf.getvalue()

        # Display the image grid
        st.image(pil_img, use_container_width=True)
        
        # Add the download button
        st.download_button(
            label="Download Image Grid",
            data=byte_im,
            file_name=f"generated_digits_{digit_to_generate}.png",
            mime="image/png"
        )
elif not generator:
    st.warning("Cannot generate images because the model failed to load.")

else:
    st.info("Adjust the controls in the sidebar and click 'Generate Images'.")