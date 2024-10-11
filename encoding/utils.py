import matplotlib.pyplot as plt
import torch
import numpy as np

plt.rcParams['figure.dpi'] = 200

# Function to display original and reconstructed images side by side
def show_image_pair(img1, img2):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(img1)
    axs[0].axis('off')
    axs[0].set_title('Original Image')
    axs[1].imshow(img2)
    axs[1].axis('off')
    axs[1].set_title('Reconstructed Image')
    plt.show()

# Function to visualize reconstructions from the VAE
def show_reconstruction(vae, data_loader, device, n_images=5):
    vae.eval()
    count = 0
    for images, in data_loader:
        images = images.to(device)
        with torch.no_grad():
            recon_images, _, _ = vae(images)
        batch_size = images.size(0)
        for i in range(batch_size):
            if count >= n_images:
                return
            original_image = images[i].cpu().numpy()
            reconstructed_image = recon_images[i].cpu().numpy()
            reconstructed_image = np.transpose(reconstructed_image, (1, 2, 0))
            # Clip the images to ensure they are in the range [0, 1]
            original_image = np.clip(original_image, 0, 1)
            reconstructed_image = np.clip(reconstructed_image, 0, 1)
            show_image_pair(original_image, reconstructed_image)
            count += 1