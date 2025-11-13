import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import itertools
from itertools import combinations_with_replacement, product
import random
from typing import List, Tuple

import numpy as np
import pandas as pd

import torch.multiprocessing as mp
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import torch
# Limit PyTorch thread usage in the parent process
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ---------------------------------------------------------------------
# Import heterogeneous network
from het_network import HetNetwork
# ---------------------------------------------------------------------

# Global config / datasets
DEVICE = torch.device("cpu")
transform = transforms.Compose([transforms.ToTensor()])

# Preload the datasets once at import.  Even though the actual DataLoader is
# built per worker, this ensures the data is available and downloaded.
TRAIN_DATASET = datasets.MNIST(root="../datasets", train=True, download=True, transform=transform)
TEST_DATASET = datasets.MNIST(root="../datasets", train=False, download=True, transform=transform)

# Shared dictionary to hold DataLoaders created in the pool initializer
_GLOBAL: dict = {}


def _pool_init(batch_size: int) -> None:
    """Pool initializer to set thread limits and build DataLoaders per worker.

    Args:
        batch_size: The batch size to use when constructing loaders.
    """
    import torch  # ensure torch is imported in subprocess
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    # Use download=False to avoid racing downloads in multiple processes
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="../datasets", train=True, download=False, transform=transform)
    test_ds = datasets.MNIST(root="../datasets", train=False, download=False, transform=transform)
    _GLOBAL["train_loader"] = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    _GLOBAL["test_loader"] = DataLoader(test_ds, batch_size=batch_size, num_workers=0)


def load_data(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Return train and test loaders created by the pool initializer.

    The initializer must have been run first; otherwise this will assert.
    """
    assert "train_loader" in _GLOBAL and "test_loader" in _GLOBAL, (
        "Pool initializer didn't run. Pass initializer=_pool_init in mp.Pool."
    )
    return _GLOBAL["train_loader"], _GLOBAL["test_loader"]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[float, float]:
    """Train the model for one epoch and return loss and accuracy."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    for data, target in loader:
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        data = data.view(data.size(0), -1)  # flatten 28×28 → 784
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        total_correct += pred.eq(target).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / len(loader.dataset)
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Tuple[float, float]:
    """Evaluate the model on a loader and return loss and accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            data = data.view(data.size(0), -1)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            total_correct += pred.eq(target).sum().item()
    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / len(loader.dataset)
    return avg_loss, accuracy


def process_activation_composition(args):
    """Evaluate one activation composition across multiple trials.

    For each trial, it randomly permutes the neuron-wise activation assignment in
    each layer, trains the model on the training set for `num_epochs` epochs,
    and then measures accuracy on the test set.
    """
    (
        activation_composition,
        architecture,
        num_epochs,
        batch_size,
        n_trials,
        base_seed,
    ) = args
    # Limit threads again inside worker
    import torch  # local import for subprocess context
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # Obtain loaders from global pool-initialized dictionary
    train_loader, test_loader = load_data(batch_size)
    input_dim = 28 * 28
    num_classes = 10
    lr = 1e-3
    criterion = nn.CrossEntropyLoss()
    test_accuracies: List[float] = []
    for trial in range(n_trials):
        # Stable seed for reproducibility across processes
        seed = (hash(activation_composition) ^ (trial + 1) ^ base_seed) & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Randomly shuffle neuron order in each layer
        activation_architecture = []
        for layer in activation_composition:
            layer_list = list(layer)
            random.shuffle(layer_list)
            activation_architecture.append(layer_list)
        model = HetNetwork(
            input_dim=input_dim,
            output_dim=num_classes,
            architecture=architecture,
            activation_funcs=activation_architecture,
        ).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=lr)
        for _ in range(num_epochs):
            train_one_epoch(model, train_loader, criterion, optimizer)
        # Evaluate once on the test set (no peeking during training)
        _, test_acc = evaluate(model, test_loader, criterion)
        test_accuracies.append(test_acc)
    return {
        "activation_composition": activation_composition,
        "architecture": architecture,
        "mean_test_accuracy": float(np.mean(test_accuracies)),
        "std_test_accuracy": float(np.std(test_accuracies)),
        "best_test_accuracy": float(np.max(test_accuracies)),
        "n_trials": int(n_trials),
        "epochs": int(num_epochs),
        "batch_size": int(batch_size),
    }


if __name__ == "__main__":
    # Use the 'spawn' start method for safety with PyTorch
    mp.set_start_method("spawn", force=True)
    # Experiment parameters
    num_epochs = 10
    batch_size = 128
    n_trials = 5
    num_processes = 12
    base_seed = 12345
    activation_functions = ["tanh", "relu", "softplus", "leaky_relu"]
    number_of_neurons = 10
    min_neurons_per_layer = 5
    min_layers = 2
    max_layers = 2
    # Enumerate architectures satisfying constraints
    architectures: List[List[int]] = []
    for n_layers in range(min_layers, max_layers + 1):
        if n_layers * min_neurons_per_layer > number_of_neurons:
            continue
        if n_layers == 1:
            architectures.append([number_of_neurons])
        else:
            remaining = number_of_neurons - n_layers * min_neurons_per_layer
            for cuts in itertools.combinations(range(remaining + n_layers - 1), n_layers - 1):
                cuts_tuple = (-1,) + cuts + (remaining + n_layers - 1,)
                extras = [cuts_tuple[i + 1] - cuts_tuple[i] - 1 for i in range(n_layers)]
                sizes = [min_neurons_per_layer + e for e in extras]
                architectures.append(sizes)
    print("Architectures:", architectures)
    # Enumerate unordered activation compositions
    activation_compositions_set = set()
    for arch in architectures:
        layer_sizes = arch
        per_layer_combos: List[List[Tuple[str, ...]]] = []
        for size in layer_sizes:
            combos = list(combinations_with_replacement(activation_functions, size))
            per_layer_combos.append([tuple(sorted(c)) for c in combos])
        for layers_combo in product(*per_layer_combos):
            activation_compositions_set.add(tuple(layers_combo))
    print(f"Testing {len(activation_compositions_set)} activation compositions across all architectures...")
    # Build job list
    jobs = []
    for comp in activation_compositions_set:
        architecture = [len(layer) for layer in comp]
        jobs.append((comp, architecture, num_epochs, batch_size, n_trials, base_seed))
    results = []
    total_jobs = len(jobs)
    # Run parallel evaluation with initializer
    if num_processes > 1:
        with mp.Pool(
            processes=num_processes,
            initializer=_pool_init,
            initargs=(batch_size,),
        ) as pool:
            completed = 0
            with tqdm(total=total_jobs) as pbar:
                chunksize = max(1, len(jobs) // (num_processes * 8))
                for res in pool.imap_unordered(process_activation_composition, jobs, chunksize=chunksize):
                    results.append(res)
                    completed += 1
                    pbar.update(1)
                    if completed % 50 == 0 or completed == total_jobs:
                        print(f"Completed {completed}/{total_jobs} jobs")
    else:
        # Serial fallback
        with tqdm(total=total_jobs) as pbar:
            for j in jobs:
                res = process_activation_composition(j)
                results.append(res)
                pbar.update(1)
    # Save results to CSV and report best configuration
    df = pd.DataFrame(results)
    out_csv = f"results_unordered_testonly_{num_epochs}.csv"
    try:
        prev = pd.read_csv(out_csv)
        df = pd.concat([prev, df], ignore_index=True)
    except FileNotFoundError:
        pass
    df.to_csv(out_csv, index=False)
    best_idx = df["best_test_accuracy"].idxmax()
    best_row = df.loc[best_idx]
    print(f"Best architecture (layer sizes): {best_row['architecture']}")
    print(
        f"Best activation composition (unordered per layer): {best_row['activation_composition']}"
    )
    print(f"Best test accuracy: {best_row['best_test_accuracy'] * 100:.2f}%")