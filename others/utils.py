import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
import torch
from torch import nn

import time
import graphviz
from PIL import Image
import io

# A simple neural network with customizable neurons and activation functions
class HetNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, number_of_neurons=5, activation_funcs=None):
        super(HetNetwork, self).__init__()

        self.neurons = nn.ModuleList([
            nn.Linear(input_dim, 1) for _ in range(number_of_neurons)
        ])

        # Map of available activation functions
        activation_functions_map = {
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'gelu': nn.GELU(),
            'softplus': nn.Softplus(),
            # Add more activation functions as needed
        }

        if activation_funcs is None:
            # Default activation functions
            activation_funcs = ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu']

        # Check if activation_funcs is a list of strings or nn.Module instances
        if all(isinstance(act, str) for act in activation_funcs):
            # Convert activation function names to instances
            try:
                self.activation_funcs = nn.ModuleList([
                    activation_functions_map[act.lower()] for act in activation_funcs
                ])
            except KeyError as e:
                raise ValueError(f"Activation function '{e.args[0]}' is not recognized. "
                                 f"Available functions are: {list(activation_functions_map.keys())}")
        elif all(isinstance(act, nn.Module) for act in activation_funcs):
            self.activation_funcs = nn.ModuleList(activation_funcs)
        else:
            raise TypeError("activation_funcs must be a list of strings or nn.Module instances.")

        if len(self.activation_funcs) != number_of_neurons:
            raise ValueError("The number of activation functions must match the number of neurons.")

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
