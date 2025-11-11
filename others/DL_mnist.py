import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.optim import Adam

import numpy as np
import itertools
from itertools import combinations_with_replacement, product

from het_network import HetNetwork
import multiprocessing
from tqdm import tqdm
import pandas as pd

transform = transforms.Compose([transforms.ToTensor()])

TRAIN_DATASET = datasets.MNIST(root='../datasets', train=True, download=True, transform=transform)
TEST_DATASET = datasets.MNIST(root='../datasets', train=False, download=True, transform=transform)
DEVICE = torch.device('cpu')

def load_data(batch_size, train_indices=None, val_indices=None):
    if train_indices is not None and val_indices is not None:
        train_dataset = Subset(TRAIN_DATASET, train_indices)
        val_dataset = Subset(TRAIN_DATASET, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        return train_loader, val_loader
    else:
        train_loader = DataLoader(TRAIN_DATASET, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TEST_DATASET, batch_size=batch_size)
        return train_loader, test_loader

def train(model, device, train_loader, criterion, optimizer):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0
    total_correct = 0

    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)

        # Flatten the data
        data = data.view(data.size(0), -1)

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        total_correct += pred.eq(target).sum().item()

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = total_correct / len(train_loader.dataset)
    return avg_loss, accuracy

def evaluate(model, device, loader, criterion):
    """
    Evaluate the model on the given dataset.
    """
    model.eval()
    total_loss = 0
    total_correct = 0

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)

            # Flatten the data
            data = data.view(data.size(0), -1)

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Accumulate metrics
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            total_correct += pred.eq(target).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / len(loader.dataset)
    return avg_loss, accuracy

def train_MNIST_model(num_neurons, activation_funcs, train_loader, test_loader, num_epochs=1):

    device = DEVICE

    # Hyperparameters
    input_dim = 28 * 28  # MNIST images are 28x28 pixels
    num_classes = 10     # MNIST has 10 classes


    model = HetNetwork(
        input_dim=input_dim,
        output_dim=num_classes,
        architecture=[num_neurons],            # single layer with num_neurons
        activation_funcs=[list(activation_funcs)]  # wrap the tuple/list into another list
    )
    model.to(device)

    learning_rate = 1e-3
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train(
            model, device, train_loader, criterion, optimizer
        )
        test_loss, test_accuracy = evaluate(
            model, device, test_loader, criterion
        )

    return test_accuracy

def process_activation_combination(args):
    """
    This function will handle one set of activation functions for all given architectures.
    """
    activation_architecture, num_epochs, batch_size, n_trials = args
    architecture = [len(layer) for layer in activation_architecture]
    activation_funcs = [act for layer in activation_architecture for act in layer]

    # Load the dataset once per call
    train_loader, test_loader = load_data(batch_size)

    results = []

    device = DEVICE
    input_dim = 28 * 28
    num_classes = 10
    learning_rate = 1e-3
    criterion = nn.CrossEntropyLoss()

    accuracies = []
    for _ in range(n_trials):
        # Setup model for this combination
        model = HetNetwork(
            input_dim=input_dim,
            output_dim=num_classes,
            architecture=architecture,
            activation_funcs=activation_architecture
        )
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train and evaluate
        for epoch in range(1, num_epochs + 1):
            _, _ = train(model, DEVICE, train_loader, criterion, optimizer)
            test_loss, test_accuracy = evaluate(model, DEVICE, test_loader, criterion)

        accuracies.append(test_accuracy)

    # Store results for this architecture and activation combo
    results.append({
        'activation_funcs': activation_architecture,
        'architecture': architecture,
        'activation_functions': activation_funcs,
        'mean_accuracy': float(np.mean(accuracies)),
        'std_accuracy': float(np.std(accuracies)),
        'best_accuracy': float(np.max(accuracies))
    })

    return results  # list of dictionaries

if __name__ == '__main__':
    num_epochs = 10
    batch_size = 128
    n_trials = 5

    activation_functions = ['sigmoid', 'tanh', 'relu', 'softplus', 'leaky_relu']

    number_of_neurons = 10          # total neurons
    min_neurons_per_layer = 4       # minimum per layer
    max_layers = 2                  # maximum number of layers you want

    architectures = []

    for n_layers in range(1, max_layers + 1):
        # skip architectures that cannot satisfy the minimum constraint
        if n_layers * min_neurons_per_layer > number_of_neurons:
            continue

        if n_layers == 1:
            # all neurons in a single layer
            architectures.append([number_of_neurons])
        else:
            # distribute the remaining neurons after giving min_neurons_per_layer to each layer
            remaining = number_of_neurons - n_layers * min_neurons_per_layer

            # compositions of "remaining" into n_layers non-negative parts
            # using stars-and-bars via combinations
            for cuts in itertools.combinations(range(remaining + n_layers - 1), n_layers - 1):
                cuts_tuple = (-1,) + cuts + (remaining + n_layers - 1,)
                extras = [cuts_tuple[i + 1] - cuts_tuple[i] - 1 for i in range(n_layers)]
                sizes = [min_neurons_per_layer + e for e in extras]
                architectures.append(sizes)

    print("Architectures:", architectures)
    # For number_of_neurons=10, min=4, max_layers=2:
    # -> [[10], [4, 6], [5, 5], [6, 4]]

    activation_architectures = set()

    for arch in architectures:
        # For each layer size in this architecture, enumerate all unordered activation multisets
        per_layer_combos = []
        for layer_size in arch:
            layer_combos = list(combinations_with_replacement(activation_functions, layer_size))
            per_layer_combos.append(layer_combos)

        # Take the Cartesian product across layers to get full activation configurations
        for layers_combo in product(*per_layer_combos):
            # layers_combo is e.g. (('relu','relu','sigmoid','sigmoid'),
            #                       ('tanh','tanh','tanh','tanh','tanh','tanh'))
            activation_architecture = [list(layer) for layer in layers_combo]
            activation_architecture_tuple = tuple(tuple(layer) for layer in activation_architecture)
            activation_architectures.add(activation_architecture_tuple)

    print(f"Testing {len(activation_architectures)} activation configurations across all architectures...")

    args_list = [
        (activation_architecture, num_epochs, batch_size, n_trials)
        for activation_architecture in activation_architectures
    ]

    num_processes = 10
    print(f'Processing using {num_processes} processes over {num_epochs} epochs and {n_trials} trials...')

    with multiprocessing.Pool(processes=num_processes) as pool:
        parallel_results = list(tqdm(
            pool.imap_unordered(process_activation_combination, args_list),
            total=len(args_list)
        ))

    # Flatten the list of lists
    flattened_results = [item for sublist in parallel_results for item in sublist]

    # Convert into a DataFrame
    result_df = pd.DataFrame(flattened_results)

    file_name = f'results_unordered_{num_epochs}.csv'
    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['activation_funcs', 'architecture', 'activation_functions', 'mean_accuracy', 'std_accuracy', 'best_accuracy', 'epochs', 'batch_size', 'num_trials', 'num_neurons'])

    result_df['epochs'] = num_epochs
    result_df['batch_size'] = batch_size
    result_df['num_trials'] = n_trials
    result_df['num_neurons'] = number_of_neurons
    df = pd.concat([df, result_df], ignore_index=True)
    df.to_csv(file_name, index=False)

    best_row = df.loc[df['best_accuracy'].idxmax()]
    print(f'Best activation function combination: {best_row["activation_functions"]}')
    print(f'Best average validation accuracy: {best_row["best_accuracy"]*100:.2f}%')
