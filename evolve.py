import os
import pickle
import neat
from multiprocessing import Pool, cpu_count
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from wandb_reporter import WandbReporter


class MNISTEvaluator:
    def __init__(self, num_workers, penalize_inactivity, num_tests):
        self.num_workers = num_workers
        self.generation = 0
        self.penalize_inactivity = penalize_inactivity
        self.num_tests = num_tests
        self.pool = Pool(processes=num_workers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load MNIST dataset
        transform = transforms.Compose([transforms.ToTensor()])
        self.train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        self.train_loader = DataLoader(self.train_data, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=64, shuffle=False)

    def eval_genomes(self, genomes, config):
        self.generation += 1
        jobs = []
        for _, genome in genomes:
            jobs.append(self.pool.apply_async(
                self.evaluate_genome,
                [genome, config]
            ))

        for job, (_, genome) in zip(jobs, genomes):
            fitness = job.get(timeout=None)
            genome.fitness = fitness

    def evaluate_genome(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters())

        # Training loop
        net.train()
        total_loss = 0
        for data, target in self.train_loader:
            data, target = data.view(-1, 28 * 28).to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate fitness as the inverse of the total loss
        fitness = -total_loss / len(self.train_loader)
        return fitness

    def __del__(self):
        self.pool.close()
        self.pool.join()
        self.pool.terminate()


def run(config_file: str, penalize_inactivity=False, num_generations=None,
        checkpoint=None, num_tests=5, num_cores=1):
    # Load the config file

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Load the population if checkpoint is not None
    pop = neat.Checkpointer.restore_checkpoint(checkpoint, config) if checkpoint is not None else neat.Population(
        config)

    # get wandbkey from wandb_key.txt file
    with open("wandb_api_key.txt", "r") as f:
        wandb_key = f.read().strip()
    save_interval = int(num_generations / 10)
    wandb_reporter = WandbReporter(wandb_key, "mnist-neat", None, save_interval, tags=["neat", "MNIST"])
    pop.add_reporter(wandb_reporter)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.Checkpointer(generation_interval=save_interval,
                                       time_interval_seconds=1800,
                                       filename_prefix="checkpoint-mnist-")
                     )

    ec = MNISTEvaluator(num_cores, penalize_inactivity, num_tests)

    print("Configuration ", pop.config.genome_config)
    # Run until the winner from a generation is able to solve the task
    gen_best = pop.run(ec.eval_genomes, num_generations)

    # Save the best model
    result_path = os.path.join(local_dir, "visualisations", "mnist")
    os.makedirs(result_path, exist_ok=True)
    with open(result_path + '/best_genome.pickle', 'wb') as f:
        pickle.dump(gen_best, f)


if __name__ == '__main__':
    run(config_file="config-mnist",
        penalize_inactivity=False,
        num_generations=100,
        num_tests=2,
        num_cores=cpu_count(),
        )