import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.optim import Adam
from itertools import product
import random
import numpy as np
from itertools import permutations

from het_network import HetNetwork
from utils import draw_het_network
import multiprocessing
from tqdm import tqdm
import time
import pandas as pd

def load_data(batch_size, train_indices=None, val_indices=None):

    transform = transforms.Compose([transforms.ToTensor(),])

    full_train_dataset = datasets.MNIST(root='../datasets', train=True, download=True, transform=transform)

    if train_indices is not None and val_indices is not None:
        # Create subsets for training and validation
        train_dataset = Subset(full_train_dataset, train_indices)
        val_dataset = Subset(full_train_dataset, val_indices)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader
    else:
        # Load the test dataset
        test_dataset = datasets.MNIST(root='../datasets', train=False, download=True, transform=transform)

        # Use the full training dataset and test dataset
        train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

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

    device = torch.device('cpu')

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

    args = (activation_functions, architectures, num_epochs, batch_size, n_trials)
    """
    activation_architecture, num_epochs, batch_size, n_trials = args
    architecture = [len(layer) for layer in activation_architecture]
    activation_funcs = [act for layer in activation_architecture for act in layer]
    # We'll record the results for every (architecture, activation combo)
    results = []

    # Load the dataset once per call
    train_loader, test_loader = load_data(batch_size)
    # Generate all permutations of the entire activation_functions list
    # If activation_functions has duplicates, permutations will include distinct orderings of those duplicates.
    accuracies = []
    # Perform multiple trials
    for i in range(n_trials):
        # Setup model for this combination
        input_dim = 28 * 28
        num_classes = 10
        model = HetNetwork(
            input_dim=input_dim,
            output_dim=num_classes,
            architecture=architecture,
            activation_funcs=activation_architecture
        )
        model.to(torch.device('cpu'))

        learning_rate = 1e-3
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train and evaluate
        for epoch in range(1, num_epochs + 1):
            train_loss, train_accuracy = train(model, torch.device('cpu'), train_loader, criterion, optimizer)
            test_loss, test_accuracy = evaluate(model, torch.device('cpu'), test_loader, criterion)

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
    activation_functions = ['sigmoid', 'tanh', 'relu', 'softplus', 'leaky_relu']
    number_of_neurons = 5
    num_epochs = 10
    batch_size = 128
    n_trials = 5

    architectures = [
        [5],  # Single layer with 5 neurons
        # [3, 2],  # Two layers: first with 3 neurons, second with 2 neurons
        # [2, 3],  # Two layers: first with 2 neurons, second with 3 neurons
        # [4, 1],
        # [1, 4],
    ]
    # Generate all combinations (as before)
    raw_combinations = product(activation_functions, repeat=number_of_neurons)

    # Convert each combination into a sorted tuple to treat it as a set (ignoring order)
    unique_combinations = set()
    for comb in raw_combinations:
        canonical_form = tuple(sorted(comb))
        unique_combinations.add(canonical_form)
    all_activation_combinations = list(unique_combinations)
    print(len(all_activation_combinations))

    hom_activation_combinations = [act_funcs for act_funcs in all_activation_combinations if len(set(act_funcs)) == 1]
    het_activation_combinations = [act_funcs for act_funcs in all_activation_combinations if len(set(act_funcs)) > 1]

    max_combinations = None
    if max_combinations is not None and len(het_activation_combinations) > max_combinations:
        het_activation_combinations = random.sample(het_activation_combinations, max_combinations)
    total_activation_combinations = hom_activation_combinations + het_activation_combinations
    print(f'Testing {len(total_activation_combinations)} of {len(all_activation_combinations)} activation function combinations...')

    activation_architectures = set()

    # Loop through each architecture
    for activation_funcs in total_activation_combinations:
        all_unique_permutations = set(permutations(activation_funcs, len(activation_funcs)))
        for architecture in architectures:
            for act_funcs_flat in all_unique_permutations:
                # Convert the flat tuple of activations into a nested list per layer
                start = 0
                activation_architecture = []
                for layer_size in architecture:
                    layer_size = int(layer_size)
                    end = start + layer_size
                    layer_acts = act_funcs_flat[start:end]
                    activation_architecture.append(list(layer_acts))
                    start = end

                activation_architecture_tuple = tuple(tuple(layer) for layer in activation_architecture)
                activation_architectures.add(activation_architecture_tuple)

    args_list = []
    for activation_architecture in activation_architectures:
        args_list.append((activation_architecture, num_epochs, batch_size, n_trials))

    num_processes = 8 # optimal seem to be around 8 (measured on M2 max for 50 combinations)
    #                Processing time
    #       2 epochs   |    10 epochs
    # -------------------------------------
    #  4 :   211.06    |     1072.92
    #  6 :   149.22    |     856.79
    #  8 :   136.34    |     759.64
    #  10:   116.30    |     816.22
    #  12:   125.13    |     689.13


    print(f'Processing using {num_processes} processes over {num_epochs} epochs and {n_trials} trials...')

    with multiprocessing.Pool(processes=num_processes) as pool:
        parallel_results = list(tqdm(pool.imap_unordered(process_activation_combination, args_list), total=len(args_list)))

    # Flatten the list of lists
    flattened_results = [item for sublist in parallel_results for item in sublist]

    # Convert into a DataFrame
    result_df = pd.DataFrame(flattened_results)

    file_name = f'results_{num_epochs}.csv'
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
    print(f'Best activation function combination: {best_row["activation_funcs"]}')
    print(f'Best average validation accuracy: {best_row["best_accuracy"]*100:.2f}%')
