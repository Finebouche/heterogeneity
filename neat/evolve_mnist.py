import os
import pickle
import multiprocessing
from wandb_reporter import WandbReporter
from neat.config_files.config_files_utils import config_to_dict

import neat
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

# Global variables to be initialized in each process
global_train_loader = None
global_val_loader = None
global_test_loader = None
global_device = None


def mnist_initializer(device, subset_size):
    global global_train_loader
    global global_val_loader
    global global_test_loader
    global global_device
    global_device = device

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Use only a subset of training data to speed up
    train_indices = torch.randperm(len(train_data))[:subset_size]
    val_indices = torch.randperm(len(train_data))[:subset_size]
    train_subset = torch.utils.data.Subset(train_data, train_indices)
    val_subset = torch.utils.data.Subset(train_data, val_indices)

    global_train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    global_val_loader = DataLoader(val_subset, batch_size=64, shuffle=True)
    global_test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


def mnist_evaluate_genome(genome, config, dataset='train'):
    global global_train_loader
    global global_val_loader
    global global_test_loader
    global global_device

    # Create network
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Select the appropriate data loader
    if dataset == 'train':
        data_loader = global_train_loader
    elif dataset == 'val':
        data_loader = global_val_loader
    elif dataset == 'test':
        data_loader = global_test_loader
    else:
        raise ValueError("dataset must be 'train', 'val' or 'test'")

    # Evaluation loop
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for data, target in data_loader:
        data = data.view(-1, 28 * 28)  # Flatten the images
        outputs = []
        for sample in data:
            output = net.activate(sample.tolist())
            outputs.append(output)

        outputs = torch.tensor(outputs)

        # Compute loss
        loss = criterion(outputs, target)
        total_loss += loss.item()

        # Get predictions
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    # Calculate fitness
    fitness = correct / total
    return fitness


def run(config_file: str, num_generations=None, checkpoint=None, num_cores=1, subset_size=1000,
        wandb_project_name=None, show_species_detail=True):
    # Load the config file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config_files", config_file)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Load the population if checkpoint is not None
    if checkpoint is not None:
        pop = neat.Checkpointer.restore_checkpoint(checkpoint, config)
    else:
        pop = neat.Population(config)

    if wandb_project_name is not None:
        with open("../wandb_api_key.txt", "r") as f:
            wandb_key = f.read().strip()

        # load the config file to pass it to wandb
        config_dict = config_to_dict(config_file)
        wandb_reporter = WandbReporter(
            project_name=wandb_project_name,
            config=config_dict,
            tags=["neat", "MNIST"],
            api_key=wandb_key  # Omit this line if you're already logged in to wandb
        )
        pop.add_reporter(wandb_reporter)

    pop.add_reporter(neat.StdOutReporter(show_species_detail))
    # pop.add_reporter(neat.Checkpointer(
    #     generation_interval=int(num_generations / 10),
    #     time_interval_seconds=1800,
    #     filename_prefix="checkpoint-mnist-"
    # ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use the custom ParallelEvaluator with initializer
    pe = neat.parallel.ParallelEvaluator(
        num_workers=num_cores,
        eval_function=mnist_evaluate_genome,
        initializer=mnist_initializer,
        initargs=(device, subset_size)
    )

    print("Configuration ", pop.config.genome_config)
    # Run the NEAT algorithm
    gen_best = pop.run(pe.evaluate, num_generations)

    # Save the best model
    result_path = os.path.join("../visualisations", "mnist")
    os.makedirs(result_path, exist_ok=True)
    with open(os.path.join(result_path, 'best_genome.pickle'), 'wb') as f:
        pickle.dump(gen_best, f)

    mnist_initializer(device, subset_size)

    # Evaluate the best genome on the test dataset
    score = mnist_evaluate_genome(gen_best, config, dataset='val')
    return score


if __name__ == '__main__':
    run(config_file="config-mnist",
        num_generations=10,
        num_cores=multiprocessing.cpu_count(),
        subset_size=1000,
        wandb_project_name="neat-mnist"
        )
