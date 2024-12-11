import gymnasium
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

from encoding.vae import load_vae_model
from het_network import HetNetwork
from utils import WandbCallback

class VAEFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, vae, vae_device):
        super(VAEFeatureExtractor, self).__init__(observation_space, features_dim=32)  # Assuming latent dim is 32
        self.vae = vae
        self.vae_device = vae_device
        self.vae.to(self.vae_device)

    def forward(self, observations):
        # observations shape: [batch_size, channels, height, width]
        observations = observations.permute(0, 2, 3, 1) # Permute to [batch_size, height, width, channels]
        observations = observations[:, 20:84, 16:80, :]  # Crop observations
        observations = observations.float() / 255.0 # Normalize
        observations = observations.to(self.vae_device) # Move to device

        # Preprocess and encode with VAE
        x = self.vae.preprocess(observations)
        with torch.no_grad():
            mu, log_var = self.vae.encoder(x)
            z = self.vae.reparameterize(mu, log_var)
            z = z.view(z.size(0), -1)  # Flatten
        return z


class IdentityMlpExtractor(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.latent_dim_pi = feature_dim
        self.latent_dim_vf = feature_dim

    def forward(self, features):
        return features, features

    def forward_actor(self, features):
        # Return features for the policy network
        return features

    def forward_critic(self, features):
        # Return features for the value network
        return features

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs
        )
        self._build_mlp_extractor()
        self._build_custom_networks()
        # Re-initialize optimizer to include new parameters
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _build_mlp_extractor(self):
        # Use the updated IdentityMlpExtractor
        self.mlp_extractor = IdentityMlpExtractor(self.features_extractor.features_dim)

    def _build_custom_networks(self):
        latent_dim = self.features_extractor.features_dim
        if isinstance(self.action_space, gymnasium.spaces.Box):
            action_dim = self.action_space.shape[0]
        else:
            action_dim = self.action_space.n

        # Initialize custom action and value networks
        self.action_net = HetNetwork(latent_dim, action_dim)
        self.value_net = HetNetwork(latent_dim, 1)

def run(env_spec_id, env_kwargs, vae_path, num_timesteps, wandb_project_name, record_video):
    # Create the environment
    env = gymnasium.make(env_spec_id, **env_kwargs)
    env = Monitor(env)

    # Load the VAE model
    vae_device = torch.device("cpu") # "cpu", "cuda" or "mps"
    device = torch.device("cpu")
    vae = load_vae_model(vae_path, vae_device)

    # Define the custom policy
    policy_kwargs = dict(
        features_extractor_class=VAEFeatureExtractor,
        features_extractor_kwargs=dict(vae_device=vae_device, vae=vae),
        # net_arch=[dict(pi=[5], vf=[5])]  # 5 hidden neurons
    )

    # Create the model
    model = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1, device=device)

    with open("../wandb_api_key.txt", "r") as f:
        wandb_key = f.read().strip()
    wandb_callback = WandbCallback(
        project_name=wandb_project_name,
        tags=[env.spec.id],
        api_key=wandb_key,
        log_network=True,
    )

    # Train the model
    model.learn(total_timesteps=int(num_timesteps), callback=wandb_callback)

    # Save the model
    model.save("ppo_vae_carracing")

    # Close the environment
    env.close()

if __name__ == '__main__':
    env_kwargs = {}  # Define any specific environment kwargs you need
    run(env_spec_id="CarRacing-v3",
        env_kwargs=env_kwargs,
        vae_path="../encoding/vae_cpu.pickle",
        num_timesteps=1e6,
        wandb_project_name="ppo-vae-carracing",
        record_video=True
        )