import wandb
from neat.reporting import BaseReporter
import numpy as np


class WandbReporter(BaseReporter):
    def __init__(self, project_name, tags=None, api_key=None):
        super().__init__()
        self.project_name = project_name
        self.tags = tags
        self.current_generation = None

        # Authenticate with wandb if api_key is provided
        if api_key is not None:
            wandb.login(key=api_key)
        # Initialize wandb run
        wandb.init(project=self.project_name, tags=self.tags)

    def start_generation(self, generation):
        self.current_generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        # Collect fitnesses
        fitnesses = [genome.fitness for genome in population.values()]
        avg_fitness = np.mean(fitnesses)
        max_fitness = np.max(fitnesses)
        min_fitness = np.min(fitnesses)
        std_fitness = np.std(fitnesses)

        # Log the statistics
        wandb.log({
            "generation": self.current_generation,
            "best_genome_fitness": best_genome.fitness,
            "avg_fitness": avg_fitness,
            "max_fitness": max_fitness,
            "min_fitness": min_fitness,
            "std_fitness": std_fitness,
            "species_count": len(species.species)
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