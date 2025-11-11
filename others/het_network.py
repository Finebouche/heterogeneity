import torch
from torch import nn
from collections import defaultdict


class HetNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, architecture, activation_funcs):
        super().__init__()

        # architecture: list of ints, neurons per layer
        # activation_funcs: list of lists (per layer) of activation names/modules

        if not isinstance(architecture, (list, tuple)):
            raise TypeError("architecture must be a list or tuple of layer sizes.")

        if not all(isinstance(layer_acts, (list, tuple)) for layer_acts in activation_funcs):
            raise TypeError("activation_funcs must be a list of lists, "
                            "one list of activations per layer.")

        if len(architecture) != len(activation_funcs):
            raise ValueError(
                "The number of activation function layers must match the number of layers in the architecture."
            )

        # Map of available activation functions -> function (not module) for speed
        activation_functions_map = {
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'relu': torch.relu,
            'leaky_relu': nn.LeakyReLU(),  # kept as module because it has a slope param
            'gelu': nn.GELU(),
            'softplus': nn.Softplus(),
        }

        self.layers = nn.ModuleList()
        # For each layer: mapping act_key -> list of neuron indices using that activation
        self.layer_act_indices = []
        # For each layer: mapping act_key -> callable (torch fn or nn.Module)
        self.layer_act_fns = []

        current_input_dim = input_dim

        for layer_idx, (layer_size, layer_acts) in enumerate(zip(architecture, activation_funcs)):
            if len(layer_acts) != layer_size:
                raise ValueError(
                    f"Layer {layer_idx} expects {layer_size} activation functions, "
                    f"but got {len(layer_acts)}."
                )

            # Single Linear for the whole layer
            linear = nn.Linear(current_input_dim, layer_size)
            self.layers.append(linear)

            # Normalize activations into callables and group neuron indices by activation
            act_indices = defaultdict(list)
            act_fns = {}

            for neuron_idx, act in enumerate(layer_acts):
                # Convert string to function / module
                if isinstance(act, str):
                    key = act.lower()
                    if key not in activation_functions_map:
                        raise ValueError(
                            f"Activation function '{act}' is not recognized. "
                            f"Available: {list(activation_functions_map.keys())}"
                        )
                    fn = activation_functions_map[key]
                elif isinstance(act, nn.Module):
                    key = str(act)  # just a label; unique enough here
                    fn = act
                else:
                    raise TypeError("Each activation must be a string or nn.Module instance.")

                act_indices[key].append(neuron_idx)
                act_fns[key] = fn

            self.layer_act_indices.append(dict(act_indices))
            self.layer_act_fns.append(act_fns)

            current_input_dim = layer_size

        # Final output layer
        self.output_layer = nn.Linear(current_input_dim, output_dim)

    def forward(self, x):
        # x: (batch, input_dim)
        for linear, act_indices, act_fns in zip(self.layers, self.layer_act_indices, self.layer_act_fns):
            # Single matmul for the layer
            z = linear(x)  # shape: (batch, layer_size)
            # Allocate output tensor
            out = torch.empty_like(z)

            # Apply each activation to its group of neurons
            for key, idxs in act_indices.items():
                fn = act_fns[key]
                idxs_tensor = torch.tensor(idxs, device=z.device)
                # select columns corresponding to these neurons
                slice_z = z[:, idxs_tensor]
                out[:, idxs_tensor] = fn(slice_z)

            x = out

        x = self.output_layer(x)
        return x