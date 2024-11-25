import torch


if __name__ == '__main__':
    from vae import VAE, train_vae
    from torch.utils.data import TensorDataset, DataLoader
    import numpy as np
    from utils import show_reconstruction
    import matplotlib.pyplot as plt
    import gymnasium as gym
    from tqdm import tqdm

    num_frame = 20

    env = gym.make('CarRacing-v3')
    frames = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Collect data with progress bar
    with tqdm(total=num_frame, desc="Collecting frames") as pbar:
        while len(frames) < num_frame:
            observation, _ = env.reset()
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

                # print the images
    for i in range(num_frame):
        plt.imshow(frames[i])
        plt.show()

    # frames = np.array(frames)  # Just convert to array, no other processing
    #
    # dataset = TensorDataset(torch.from_numpy(frames).float())  # Ensure data is float when creating the dataset
    # data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    #
    # vae = VAE(32)
    # vae.load_state_dict(torch.load('vae.pickle', weights_only=True))
    #
    # for parameter in vae.parameters():
    #     print(parameter.shape)
    #
    # # calculate the number of parameters
    # num_params = sum(p.numel() for p in vae.parameters())
    # print(f"Number of parameters: {num_params}")
    #
    # show_reconstruction(vae, data_loader, device, n_images=5)
