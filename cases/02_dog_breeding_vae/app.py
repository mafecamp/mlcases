from src.model import default_vae
from src.images import loadImage, reconstruct, morphBetweenImages
import matplotlib.pyplot as plt
import argparse

# silence tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(
    prog="dog breeding VAE", description="App de criação de imagens de cachorros."
)

parser.add_argument("--dog1", type=str, help="Path to the first dog image", required=True)
parser.add_argument("--dog2", type=str, help="Path to the second dog image", required=True)
parser.add_argument("--model", type=str, help="Path to the trained model", default="models/50epochs")
parser.add_argument("--mode", type=str, help="Run mode", default="blend")
parser.add_argument("--show", type=bool, help="Show image", default=False)
parser.add_argument("--save-path", type=str, help="Path to save the image (optional)", default=None)

parsed_args = parser.parse_args()

dog_1_path = parsed_args.dog1
dog_2_path = parsed_args.dog2
model_path = parsed_args.model
mode = parsed_args.mode
show_img = parsed_args.show
save_path = parsed_args.save_path

# Instancia o modelo
VAE = default_vae()
R_LOSS_FACTOR = 1000

VAE.load_trained_model(load_path=model_path, r_loss_factor=R_LOSS_FACTOR)

dog_1 = loadImage(dog_1_path)
dog_2 = loadImage(dog_2_path)

# Caso de uso 1: Reconstrução da imagem
if mode == "reconstruct":
    reconstructed_dog_1 = reconstruct(VAE, dog_1)

    # show image
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(dog_1)
    ax[0].set_title("Original")

    ax[1].imshow(reconstructed_dog_1[0])
    ax[1].set_title("Reconstructed")

    plt.show()

# Caso de uso 2: Blend de imagens
if mode == "blend":
    dogs_plot = morphBetweenImages(dog_1, dog_2, VAE, 10)

    if save_path:
        dogs_plot.savefig(save_path)

    if show_img:
        dogs_plot.show()
