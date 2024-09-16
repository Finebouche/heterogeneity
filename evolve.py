import os
import pickle
import neat
import multiprocessing
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from wandb_reporter import WandbReporter

# Global variables to be initialized in each process
global_train_loader = None
global_device = None


def initializer(device, subset_size):
    global global_train_loader
    global global_device
    global_device = device

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Use only a subset of data for evaluation to speed up
    indices = torch.randperm(len(train_data))[:subset_size]
    train_subset = torch.utils.data.Subset(train_data, indices)
    global_train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)


def evaluate_genome(genome, config):
    global global_train_loader
    global global_device

    # Create network
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Evaluation loop
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    for data, target in global_train_loader:
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
    accuracy = correct / total
    fitness = accuracy  # Or use -total_loss / len(global_train_loader)
    return fitness


# Custom ParallelEvaluator with initializer
class MyParallelEvaluator(object):
    def __init__(self, num_workers, eval_function, timeout=None, initializer=None, initargs=()):
        self.eval_function = eval_function
        self.timeout = timeout
        self.pool = multiprocessing.Pool(processes=num_workers, initializer=initializer, initargs=initargs)

    def evaluate(self, genomes, config):
        jobs = []
        for genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config)))

        for job, (genome_id, genome) in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)


def run(config_file: str, penalize_inactivity=False, num_generations=None,
        checkpoint=None, num_tests=5, num_cores=1, subset_size=1000):
    # Load the config file
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Load the population if checkpoint is not None
    if checkpoint is not None:
        pop = neat.Checkpointer.restore_checkpoint(checkpoint, config)
    else:
        pop = neat.Population(config)

    # Read wandb API key from file (if needed)
    with open("wandb_api_key.txt", "r") as f:
        wandb_key = f.read().strip()

    # Initialize WandbReporter correctly
    wandb_reporter = WandbReporter(
        project_name="mnist-neat",
        tags=["neat", "MNIST"],
        api_key=wandb_key  # Omit this line if you're already logged in to wandb
    )

    pop.add_reporter(wandb_reporter)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(
        generation_interval=int(num_generations / 10),
        time_interval_seconds=1800,
        filename_prefix="checkpoint-mnist-"
    ))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use the custom ParallelEvaluator with initializer
    pe = MyParallelEvaluator(
        num_workers=num_cores,
        eval_function=evaluate_genome,
        initializer=initializer,
        initargs=(device, subset_size)
    )

    print("Configuration ", pop.config.genome_config)
    # Run until the winner from a generation is able to solve the task
    gen_best = pop.run(pe.evaluate, num_generations)

    # Save the best model
    result_path = os.path.join("visualisations", "mnist")
    os.makedirs(result_path, exist_ok=True)
    with open(os.path.join(result_path, 'best_genome.pickle'), 'wb') as f:
        pickle.dump(gen_best, f)


if __name__ == '__main__':
    run(config_file="config-mnist",
        penalize_inactivity=False,
        num_generations=100,
        num_tests=2,
        num_cores=multiprocessing.cpu_count(),
        subset_size=1000  # Adjust as needed
        )
