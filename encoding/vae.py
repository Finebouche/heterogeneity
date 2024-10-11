import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np
from tqdm import tqdm
import gymnasium as gym


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

# Function to collect and resize frames from the environment
def generate_dataset(env_name='CarRacing-v3', num_frame=10000):
    # Setup the environment
    env = gym.make(env_name)
    frames = []

    # Collect data with progress bar
    with tqdm(total=num_frame, desc="Collecting frames") as pbar:
        while len(frames) < num_frame:
            observation, _ = env.reset()
            # Resize observation to 64x64 by croping the bottom part and the two sides
            observation = observation[20:84, 16:80, :]
            frames.append(observation)
            pbar.update(1)
            done = False
            while not done and len(frames) < num_frame:
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                observation = observation[20:84, 16:80, :]
                frames.append(observation)
                pbar.update(1)  # Update the progress bar
                done = terminated or truncated

    env.close()
    return frames

# Convolutional Encoder
class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2)    # Output: 32x31x31
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)   # Output: 64x14x14
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)  # Output: 128x6x6
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2) # Output: 256x2x2
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dims)
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dims)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # x shape: N x 32 x 31 x 31
        x = F.relu(self.conv2(x))   # x shape: N x 64 x 14 x 14
        x = F.relu(self.conv3(x))   # x shape: N x 128 x 6 x 6
        x = F.relu(self.conv4(x))   # x shape: N x 256 x 2 x 2
        x = x.contiguous().view(x.size(0), -1)   # Flatten
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var

# Convolutional Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dims, 256 * 2 * 2)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2)  # Output: 128x5x5
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2)   # Output: 64x13x13
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2)    # Output: 32x30x30
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2)     # Output: 3x64x64

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 1024, 1, 1)
        x = F.relu(self.deconv1(x))   # x shape: N x 128 x 5 x 5
        x = F.relu(self.deconv2(x))   # x shape: N x 64 x 13 x 13
        x = F.relu(self.deconv3(x))   # x shape: N x 32 x 30 x 30
        x = torch.sigmoid(self.deconv4(x))  # x shape: N x 3 x 64 x 64
        return x

# Variational Autoencoder combining the encoder and decoder
class VAE(nn.Module):
    def __init__(self, latent_dims):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def preprocess(self, x):
        # Normalize and reshape input
        if x.dim() == 3:  # Single image (H, W, C)
            x = x.unsqueeze(0)  # Make it (1, H, W, C) to simulate a batch of size 1
        x = x.float() / 255.0  # Normalize images to [0, 1]
        x = x.permute(0, 3, 1, 2)  # Transpose from (N, H, W, C) to (N, C, H, W)
        return x

    def forward(self, x):
        x = self.preprocess(x)
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

# Training function for the VAE
def train_vae(vae, data_loader, epochs=1):
    optimizer = torch.optim.Adam(vae.parameters())
    vae.train()

    # Outer loop for epochs
    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Inner loop for batches
        for x_batch, in tqdm(data_loader, desc=f"Training Epoch {epoch + 1}", leave=False):
            x = x_batch.to(device)
            optimizer.zero_grad()
            x_hat, mu, log_var = vae(x)

            # Ensure that x is in the correct shape/format before loss calculation
            x_preprocessed = vae.preprocess(x)  # We apply preprocess to ensure compatibility in the loss function

            # Reconstruction loss: Note x is already preprocessed in forward, so use x_preprocessed for correct comparison
            recon_loss = F.mse_loss(x_hat, x_preprocessed, reduction='sum')
            # KL divergence
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            # Total loss
            loss = recon_loss + kl_div
            loss.backward()
            optimizer.step()

    return vae

# Main block
if __name__ == '__main__':
    from torch.utils.data import TensorDataset, DataLoader
    from utils import show_reconstruction

    # Collect and preprocess data
    frames = generate_dataset(num_frame=10000)
    frames = np.array(frames)  # Just convert to array, no other processing

    # Create a TensorDataset without additional preprocessing
    dataset = TensorDataset(torch.from_numpy(frames))
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Instantiate the VAE model
    latent_dims = 32  # Set Nz to 32 for the Car Racing task
    vae = VAE(latent_dims).to(device)

    # Train the VAE
    vae = train_vae(vae, data_loader, epochs=20)

    # Visualize reconstructions
    show_reconstruction(vae, data_loader, device, n_images=5)

    # Save the trained model
    torch.save(vae.state_dict(), 'vae.pickle')