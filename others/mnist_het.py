import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.optim import Adam
from itertools import product
import random
import numpy as np
from sklearn.model_selection import KFold
from utils import draw_het_network, HetNetwork
import multiprocessing

def load_data(batch_size, train_indices=None, val_indices=None):
    """
    Load the MNIST dataset and return data loaders for training and validation.
    If train_indices and val_indices are None, load the full training dataset and test dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load the full training dataset
    full_train_dataset = datasets.MNIST(
        root='../datasets', train=True, download=True, transform=transform
    )

    # Load the test dataset
    test_dataset = datasets.MNIST(
        root='../datasets', train=False, download=True, transform=transform
    )

    if train_indices is not None and val_indices is not None:
        # Create subsets for training and validation
        train_dataset = Subset(full_train_dataset, train_indices)
        val_dataset = Subset(full_train_dataset, val_indices)

        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size
        )

        return train_loader, val_loader
    else:
        # Use the full training dataset and test dataset
        train_loader = DataLoader(
            full_train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size
        )

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

def train_MNIST_model(num_neurons, activation_funcs, train_loader, val_loader, num_epochs=1):
    """
    Train and evaluate the MNIST model with specified activation functions.
    """
    # Set device
    device = torch.device('cpu')

    # Hyperparameters
    input_dim = 28 * 28  # MNIST images are 28x28 pixels
    num_classes = 10     # MNIST has 10 classes

    learning_rate = 1e-3

    # Initialize model, criterion, and optimizer
    model = HetNetwork(
        input_dim=input_dim,
        output_dim=num_classes,
        number_of_neurons=num_neurons,
        activation_funcs=activation_funcs
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train(
            model, device, train_loader, criterion, optimizer
        )
        val_loss, val_accuracy = evaluate(
            model, device, val_loader, criterion
        )

    return val_accuracy

def process_activation_combination(args):
    activation_funcs, number_of_neurons, num_epochs, batch_size, n_splits = args
    # print(f"Processing : {activation_funcs}", flush=True)  # Added print statement
    fold_accuracies = []

    # Load the full dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    full_dataset = datasets.MNIST(
        root='../datasets', train=True, download=True, transform=transform
    )
    dataset_size = len(full_dataset)
    indices = np.arange(dataset_size)

    # Set up K-fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True)

    for fold, (train_indices, val_indices) in enumerate(kf.split(indices)):
        # Load data for this fold
        train_loader, val_loader = load_data(batch_size, train_indices, val_indices)
        val_accuracy = train_MNIST_model(
            number_of_neurons, activation_funcs, train_loader, val_loader, num_epochs=num_epochs
        )
        fold_accuracies.append(val_accuracy)

    avg_accuracy = sum(fold_accuracies) / n_splits
    return (activation_funcs, avg_accuracy)

if __name__ == '__main__':
    activation_functions = ['sigmoid', 'tanh', 'relu', 'softplus', 'leaky_relu']
    number_of_neurons = 5
    num_epochs = 2
    batch_size = 128
    n_splits = 5

    # Generate all possible combinations of activation functions
    activation_combinations = list(product(activation_functions, repeat=number_of_neurons))

    # Optionally, limit the number of combinations for faster execution
    max_combinations = 50
    if len(activation_combinations) > max_combinations:
        activation_combinations = random.sample(activation_combinations, max_combinations)

    # Prepare arguments for multiprocessing
    args_list = [
        (activation_funcs, number_of_neurons, num_epochs, batch_size, n_splits)
        for activation_funcs in activation_combinations
    ]

    num_processes = 6 # optimal seem to be around 6 (measured on M2 max)
    # Processing time for 4: 211.06 seconds
    # Processing time for 6: 149.22 seconds
    # Processing time for 8: 136.34 seconds
    # Processing time for 10: 116.30 seconds
    # Processing time for 12: 125.13 seconds

    # Use multiprocessing Pool to parallelize the computation
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_activation_combination, args_list)

    # Collect the best combination
    best_accuracy = 0.0
    best_combination = None
    for activation_funcs, avg_accuracy in results:
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_combination = activation_funcs

    print(f'Best activation function combination: {best_combination}')
    print(f'Best average validation accuracy: {best_accuracy*100:.2f}%')