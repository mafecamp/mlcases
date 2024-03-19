from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def loadImage(image_path, INPUT_DIMS=(128, 128, 3)):
    pil_image = Image.open(image_path)
    pil_image = pil_image.resize(
        (INPUT_DIMS[0], INPUT_DIMS[1]), Image.Resampling.LANCZOS
    )
    image = np.array(pil_image) / 255.0
    return image


def reconstruct(VAE_model, image):
    latent_space = VAE_model.encoder.predict(np.expand_dims(image, 0))
    reconstructed_image = VAE_model.decoder.predict(latent_space)
    return reconstructed_image


def morphBetweenImages(img1, img2, VAE, num_of_morphs):
    # load images
    # img1 = loadImage(images_path, example_ind1)
    # img2 = loadImage(images_path, example_ind2)
    # define alpha
    alpha = np.linspace(0, 1, num_of_morphs)
    # get latent spaces
    z1 = VAE.encoder.predict(np.expand_dims(img1, 0))
    z2 = VAE.encoder.predict(np.expand_dims(img2, 0))
    # morph and plot
    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(1, num_of_morphs + 2, 1)
    ax.imshow(img1)
    ax.axis("off")
    ax.set_title(loc="center", label="original image 1", fontsize=10)
    for i in range(num_of_morphs):
        z = z1 * (1 - alpha[i]) + z2 * alpha[i]
        new_img = VAE.decoder.predict(z)
        ax = fig.add_subplot(1, num_of_morphs + 2, i + 2)
        ax.imshow(new_img.squeeze())
        ax.axis("off")
        ax.set_title(loc="center", label="alpha={:.2f}".format(alpha[i]))
    ax = fig.add_subplot(1, num_of_morphs + 2, num_of_morphs + 2)
    ax.imshow(img2)
    ax.axis("off")
    ax.set_title(loc="center", label="original image 2", fontsize=10)

    plt.show()
