import itertools
from itertools import combinations_with_replacement, product
import random
from typing import List, Tuple  # Sequence removed

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ---------------------------------------------------------------------
# Import heterogeneous network
from het_network import HetNetwork
# ---------------------------------------------------------------------

# Global config / datasets
DEVICE = torch.device("cpu")
transform = transforms.Compose([transforms.ToTensor()])

TRAIN_DATASET = datasets.MNIST(root="../datasets", train=True, download=True, transform=transform)
TEST_DATASET = datasets.MNIST(root="../datasets", train=False, download=True, transform=transform)

def load_data(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """
    Returns train and test loaders. No validation split is used.
    """
    train_loader = DataLoader(TRAIN_DATASET, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TEST_DATASET, batch_size=batch_size)
    return train_loader, test_loader

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    for data, target in loader:
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        data = data.view(data.size(0), -1)  # flatten 28x28 -> 784

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

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
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
    """
    Evaluate one unordered activation composition across multiple trials.
    On each trial, the per-neuron order is randomly shuffled layer-wise before training,
    so we don't privilege a particular ordering but also avoid enumerating all permutations.
    """
    (
        activation_composition,  # tuple of layers; each layer is a tuple of activation names (sorted)
        architecture,            # list of ints; e.g., [6, 6]
        num_epochs,
        batch_size,
        n_trials,
        base_seed,
    ) = args

    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    train_loader, test_loader = load_data(batch_size)

    input_dim = 28 * 28
    num_classes = 10
    lr = 1e-3
    criterion = nn.CrossEntropyLoss()

    test_accuracies: List[float] = []

    for trial in range(n_trials):
        # Seed per trial for reproducibility (works across processes)
        seed = (hash(activation_composition) ^ (trial + 1) ^ base_seed) & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Randomly shuffle neuron-wise order per layer (keeps multiplicity, ignores order)
        activation_architecture = []
        for layer in activation_composition:
            layer_list = list(layer)
            random.shuffle(layer_list)  # in-place shuffle preserving duplicates
            activation_architecture.append(layer_list)

        # Build and train the model
        model = HetNetwork(
            input_dim=input_dim,
            output_dim=num_classes,
            architecture=architecture,
            activation_funcs=activation_architecture
        ).to(DEVICE)

        optimizer = Adam(model.parameters(), lr=lr)

        for _ in range(num_epochs):
            train_one_epoch(model, train_loader, criterion, optimizer)

        # Evaluate exactly once on the test set (no test peeking during training)
        _, test_acc = evaluate(model, test_loader, criterion)
        test_accuracies.append(test_acc)

    return {
        "activation_composition": activation_composition,  # unordered (per layer)
        "architecture": architecture,
        "mean_test_accuracy": float(np.mean(test_accuracies)),
        "std_test_accuracy": float(np.std(test_accuracies)),
        "best_test_accuracy": float(np.max(test_accuracies)),
        "n_trials": int(n_trials),
        "epochs": int(num_epochs),
        "batch_size": int(batch_size),
    }

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # -------------------- Experiment params --------------------
    num_epochs = 10
    batch_size = 128
    n_trials = 5
    num_processes = 12
    base_seed = 12345  # for reproducibility across runs

    activation_functions = ["tanh", "relu", "softplus", "leaky_relu"]
    number_of_neurons = 10        # total neurons across hidden layers
    min_neurons_per_layer = 5     # minimum per layer
    min_layers = 2
    max_layers = 2

    # -------------------- Enumerate architectures --------------------
    architectures: List[List[int]] = []
    for n_layers in range(min_layers, max_layers + 1):
        if n_layers * min_neurons_per_layer > number_of_neurons:
            continue
        if n_layers == 1:
            architectures.append([number_of_neurons])
        else:
            # Distribute the remaining neurons after giving min_per_layer to each layer (stars and bars)
            remaining = number_of_neurons - n_layers * min_neurons_per_layer
            for cuts in itertools.combinations(range(remaining + n_layers - 1), n_layers - 1):
                cuts_tuple = (-1,) + cuts + (remaining + n_layers - 1,)
                extras = [cuts_tuple[i + 1] - cuts_tuple[i] - 1 for i in range(n_layers)]
                sizes = [min_neurons_per_layer + e for e in extras]
                architectures.append(sizes)

    print("Architectures:", architectures)

    # -------------------- Enumerate unordered activation compositions (inline) --------------------
    activation_compositions_set = set()
    for arch in architectures:
        layer_sizes = arch
        # Build per-layer unordered multisets of activations
        per_layer_combos: List[List[Tuple[str, ...]]] = []
        for size in layer_sizes:
            # combinations_with_replacement already returns non-decreasing tuples
            combos = list(combinations_with_replacement(activation_functions, size))
            # store each layer multiset as a sorted tuple (canonical form)
            per_layer_combos.append([tuple(sorted(c)) for c in combos])
        # Cartesian product across layers -> full compositions (unordered per layer)
        for layers_combo in product(*per_layer_combos):
            activation_compositions_set.add(tuple(layers_combo))

    print(f"Testing {len(activation_compositions_set)} activation compositions across all architectures...")

    # -------------------- Build job list --------------------
    jobs = []
    for comp in activation_compositions_set:
        architecture = [len(layer) for layer in comp]
        jobs.append((comp, architecture, num_epochs, batch_size, n_trials, base_seed))

    # -------------------- Run (possibly in parallel) --------------------
    results = []
    total_jobs = len(jobs)
    if num_processes > 1:
        with mp.Pool(processes=num_processes) as pool:
            completed = 0
            with tqdm(total=total_jobs) as pbar:
                # imap_unordered yields results as soon as each worker finishes a job
                for res in pool.imap_unordered(process_activation_composition, jobs, chunksize=1):
                    results.append(res)
                    completed += 1
                    pbar.update(1)  # update tqdm bar
                    # Optional: print every N completions
                    if completed % 50 == 0 or completed == total_jobs:
                        print(f"Completed {completed}/{total_jobs} jobs")
    else:
        results = []
        total_jobs = len(jobs)
        with tqdm(total=total_jobs) as pbar:
            for j in jobs:
                res = process_activation_composition(j)
                results.append(res)
                pbar.update(1)

    # -------------------- Save + report --------------------
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
    print(f"Best activation composition (unordered per layer): {best_row['activation_composition']}")
    print(f"Best test accuracy: {best_row['best_test_accuracy']*100:.2f}%")