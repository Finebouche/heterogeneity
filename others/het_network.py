import torch
from torch import nn


class HetNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, architecture, activation_funcs):
        super(HetNetwork, self).__init__()

        # architecture: a list of integers, each representing the number of neurons in a layer
        # activation_funcs: a list of lists, each inner list corresponding to a layer
        # and containing activation function names/modules for each neuron in that layer.

        if not isinstance(architecture, (list, tuple)):
            raise TypeError("architecture must be a list or tuple of layer sizes.")

        # Check that activation_funcs is a list of lists
        if not all(isinstance(layer_acts, (list, tuple)) for layer_acts in activation_funcs):
            raise TypeError("activation_funcs must be a list of lists, where each sub-list "
                            "corresponds to a layer.")

        if len(architecture) != len(activation_funcs):
            raise ValueError(
                "The number of activation function layers must match the number of layers in the architecture.")

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

        # Convert architecture into a list of layers (ModuleList of neurons and activation funcs)
        self.layers = nn.ModuleList()
        self.activations = []

        # Keep track of the input dimension to the current layer
        current_input_dim = input_dim

        for layer_idx, (layer_size, layer_acts) in enumerate(zip(architecture, activation_funcs)):

            # Check that layer_acts matches layer_size
            if len(layer_acts) != layer_size:
                raise ValueError(f"Layer {layer_idx} expects {layer_size} activation functions, "
                                 f"but got {len(layer_acts)}.")

            # Create the neurons (Linear layers) for this layer
            layer_neurons = nn.ModuleList([nn.Linear(current_input_dim, 1) for _ in range(layer_size)])
            self.layers.append(layer_neurons)

            # Convert activation functions from strings/modules to a ModuleList
            if all(isinstance(act, str) for act in layer_acts):
                try:
                    layer_activation_funcs = nn.ModuleList(
                        [activation_functions_map[act.lower()] for act in layer_acts])
                except KeyError as e:
                    raise ValueError(f"Activation function '{e.args[0]}' is not recognized. "
                                     f"Available functions are: {list(activation_functions_map.keys())}")
            elif all(isinstance(act, nn.Module) for act in layer_acts):
                layer_activation_funcs = nn.ModuleList(layer_acts)
            else:
                raise TypeError("All activation functions in a layer must be strings or nn.Module instances.")

            self.activations.append(layer_activation_funcs)

            # After this layer, the output dimension is layer_size
            current_input_dim = layer_size

        # Create the final output layer (maps from last layer size to output_dim)
        self.output_layer = nn.Linear(current_input_dim, output_dim)

    def forward(self, x):
        # Forward pass through each layer
        for layer_neurons, layer_acts in zip(self.layers, self.activations):
            # Apply each neuron and activation in the layer
            layer_outputs = []
            for neuron, activation in zip(layer_neurons, layer_acts):
                out = activation(neuron(x))
                layer_outputs.append(out)
            # Concatenate the outputs of all neurons in this layer
            x = torch.cat(layer_outputs, dim=1)

        # After all hidden layers, apply the final output layer
        x = self.output_layer(x)
        return x