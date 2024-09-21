import wandb
from neat.reporting import BaseReporter
import numpy as np
import neat

class WandbReporter(BaseReporter):
    def __init__(self, project_name, config, tags=None, api_key=None):
        super().__init__()
        self.project_name = project_name
        self.config = config
        self.tags = tags
        self.current_generation = None

        # Authenticate with wandb if api_key is provided
        if api_key is not None:
            wandb.login(key=api_key)
        # Initialize wandb run
        wandb.init(project=self.project_name, config=self.config, tags=self.tags)

    def start_generation(self, generation):
        self.current_generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        # Collect fitnesses
        fitnesses = [genome.fitness for genome in population.values()]
        avg_fitness = np.mean(fitnesses)
        max_fitness = np.max(fitnesses)
        min_fitness = np.min(fitnesses)
        std_fitness = np.std(fitnesses)

        network_sizes = [len(genome.nodes) for genome in population.values()]
        avg_network_size = np.mean(network_sizes)
        max_network_size = np.max(network_sizes)
        min_network_size = np.min(network_sizes)
        std_network_size = np.std(network_sizes)

        # Log the statistics
        wandb.log({
            "generation": self.current_generation,
            "species_count": len(species.species),
            "fitness/best_genome_fitness": best_genome.fitness,
            "fitness/avg": avg_fitness,
            "fitness/max": max_fitness,
            "fitness/min": min_fitness,
            "fitness/std": std_fitness,
            "network_size/avg": avg_network_size,
            "network_size/max": max_network_size,
            "network_size/min": min_network_size,
            "network_size/std": std_network_size,
        })

    def end_generation(self, config, population, species_set):
        pass

    def post_reproduction(self, config, population, species):
        pass

    def complete_extinction(self):
        pass

    def found_solution(self, config, generation, best):
        pass

    def species_stagnant(self, sid, species):
        pass

    def info(self, msg):
        pass