import torch
from model import Generator

def generate_digit_images(digit, num_images=5):
    device = torch.device("cpu")
    latent_dim = 100

    # Load model
    model = Generator(nz=latent_dim)
    model.load_state_dict(torch.load("generator.pt", map_location=device))
    model.eval()

    # Generate inputs
    z = torch.randn(num_images, latent_dim)
    labels = torch.full((num_images,), digit, dtype=torch.long)

    with torch.no_grad():
        images = model(z, labels)

    return images
