import gymnasium
import torch
import numpy as np
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn
import time
import graphviz
from PIL import Image
import io

from encoding.vae import load_vae_model

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


# A simple neural network with 5 neurons and different activation functions
class HetNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, number_of_neurons=5, activation_funcs=None):
        super(HetNetwork, self).__init__()

        self.neurons = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(number_of_neurons)
        ])

        if activation_funcs is None:
            # Default activation functions
            activation_funcs = [nn.ReLU(), nn.Tanh(), nn.Sigmoid(), nn.LeakyReLU(), nn.ELU()]
        self.activation_funcs = nn.ModuleList(activation_funcs)

        self.output_layer = nn.Linear(number_of_neurons, output_dim)

    def forward(self, x):
        # Apply each neuron with its activation function
        outputs = []
        for neuron, activation in zip(self.neurons, self.activation_funcs):
            out = activation(neuron(x))
            outputs.append(out)
        # Concatenate the outputs
        out = torch.cat(outputs, dim=1)
        # Pass through the output layer
        out = self.output_layer(out)
        return out

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

def draw_het_network(model, format='png'):

    node_attrs = {
        'shape': 'circle',
        'fontsize': '10',
        'height': '0.1',
        'width': '0.1',
        'label': '',
        'fixedsize': 'true'
    }

    graph_attrs = {
        'rankdir': 'LR',          # Left to right direction
        'ranksep': '3.0',         # Increase the distance between ranks
        'nodesep': '0.1',         # Increase the distance between nodes
        'outputorder': 'edgesfirst',  # Draw edges first
    }


    # Create a Digraph
    dot = graphviz.Digraph(format=format, node_attr=node_attrs, graph_attr=graph_attrs    )

    # Input layer nodes
    input_dim = model.neurons[0].in_features
    for i in range(input_dim):
        dot.node(f'i{i}', xlabel=f'Input {i}', shape='circle', style='filled', fillcolor='lightgray')

    # Hidden layer nodes with activation functions
    for idx, activation in enumerate(model.activation_funcs):
        activation_name = activation.__class__.__name__
        dot.node(f'h{idx}', xlabel=f'{activation_name}', shape='circle', style='filled', fillcolor='lightblue')

    # Output layer nodes
    output_dim = model.output_layer.out_features
    for o in range(output_dim):
        dot.node(f'o{o}', xlabel=f'Output {o}', shape='circle', style='filled', fillcolor='lightgreen')

    # Edges from inputs to hidden neurons
    for idx, neuron in enumerate(model.neurons):
        weights = neuron.weight.data.squeeze().tolist()
        if input_dim == 1:
            weights = [weights]
        for i, weight in enumerate(weights):
            edge_attrs = {
                'style': 'solid',
                'color': 'lightblue',
                'penwidth': str(0.1 + abs(weight / 3.0)),
                'arrowhead': 'none'  # No arrowhead on the edge
            }

            dot.edge(f'i{i}', f'h{idx}', label=f'{weight:.2f}', _attributes=edge_attrs)

    # Edges from hidden neurons to outputs
    output_weights = model.output_layer.weight.data
    for o in range(output_dim):
        for idx in range(5):
            weight = output_weights[o, idx].item()
            edge_attrs = {
                'style': 'solid',
                'color': 'lightblue',
                'penwidth': str(0.1 + abs(weight * 3.0)),
                'arrowhead': 'none'  # No arrowhead on the edge
            }
            dot.edge(f'h{idx}', f'o{o}', label=f'{weight:.2f}', _attributes=edge_attrs)

    # Return the rendered image data
    dot_data = dot.pipe(format=format)
    return dot_data

class WandbCallback(BaseCallback):
    def __init__(self, project_name, tags, api_key, log_network, video_log_function=None):
        super().__init__()
        self.project_name = project_name
        self.log_network = log_network
        self.tags = tags
        self.current_generation = None
        self.video_log_function = video_log_function
        # Initialize variables for tracking
        self.start_time = time.time()
        self.iterations = 0

        # Authenticate with wandb if api_key is provided
        if api_key is not None:
            wandb.login(key=api_key)

        # Initialize wandb run
        wandb.init(project=self.project_name, tags=self.tags)


    def _on_step(self) -> bool:
        return True


    def _on_rollout_end(self):
        # Increment the iteration counter
        self.iterations += 1
        # Calculate time elapsed
        time_elapsed = time.time() - self.start_time
        # Calculate frames per second (fps)
        fps = self.num_timesteps / time_elapsed if time_elapsed > 0 else float('inf')

        # Log metrics to wandb
        if len(self.model.ep_info_buffer) > 0:
            rewards = [ep_info['r'] for ep_info in self.model.ep_info_buffer]
            lengths = [ep_info['l'] for ep_info in self.model.ep_info_buffer]

            log_data = {
                'episode/reward_mean': np.mean(rewards),
                'episode/length_mean': np.mean(lengths),
                'time/fps': fps,
                'time/iterations': self.iterations,
                'time/time_elapsed': time_elapsed,
                'time/total_timesteps': self.num_timesteps,
            }

            # Log network visualization
            if self.log_network:
                # Call draw_net on the action network of the policy
                image_data = draw_het_network(self.model.policy.action_net, format='png')
                image = Image.open(io.BytesIO(image_data))
                log_data['network'] = wandb.Image(image)

            wandb.log(log_data)

        return True

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

    with open("wandb_api_key.txt", "r") as f:
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
        vae_path="encoding/vae_cpu.pickle",
        num_timesteps=1e6,
        wandb_project_name="ppo-vae-carracing",
        record_video=True
    )