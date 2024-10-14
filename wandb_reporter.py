import wandb
from neat.reporting import BaseReporter
import numpy as np
from visualization import draw_net
import io
from PIL import Image

class WandbReporter(BaseReporter):
    def __init__(self, project_name, tags, api_key, log_config, log_network, video_log_function=None):
        super().__init__()
        self.project_name = project_name
        self.log_config = log_config
        self.log_network = log_network
        self.tags = tags
        self.current_generation = None
        self.video_log_function = video_log_function

        # Authenticate with wandb if api_key is provided
        if api_key is not None:
            wandb.login(key=api_key)

        # Initialize wandb run
        if self.log_config is not None:
            wandb.init(project=self.project_name, config=self.log_config, tags=self.tags)
        else:
            wandb.init(project=self.project_name, tags=self.tags)

    def start_generation(self, generation):
        self.current_generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        fitnesses = [genome.fitness for genome in population.values()]
        avg_fitness = np.mean(fitnesses)
        std_fitness = np.std(fitnesses)

        network_sizes = [len(genome.nodes) for genome in population.values()]
        avg_network_size = np.mean(network_sizes)
        max_network_size = np.max(network_sizes)
        min_network_size = np.min(network_sizes)
        std_network_size = np.std(network_sizes)

        # Prepare the data to log
        log_data = {
            "generation": self.current_generation,
            "species_count": len(species.species),
            "best_genome/fitness": best_genome.fitness,
            "best_genome/network_size": len(best_genome.nodes),
            "fitness/avg": avg_fitness,
            "fitness/std": std_fitness,
            "network_size/avg": avg_network_size,
            "network_size/max": max_network_size,
            "network_size/min": min_network_size,
            "network_size/std": std_network_size,
        }

        if self.video_log_function is not None:
            numpy_array_video = self.video_log_function(self.current_generation, best_genome, config)
            if numpy_array_video is not None:
                log_data["video"] = wandb.Video(numpy_array_video, fps=15, format="gif")

        if self.log_network:
            # Log the best genome network
            network_drawing = draw_net(config, best_genome, view=False, fmt="png")
            # Convert bytes to a PIL Image
            network_image = Image.open(io.BytesIO(network_drawing))
            log_data["neural_network_graph"] = wandb.Image(network_image)

        # Log all data at once
        wandb.log(log_data)

    def end_generation(self, config, population, species_set):
        pass

    def post_reproduction(self, config, population, species):
        pass

    def complete_extinction(self):
        pass

    def found_solution(self, config, generation, best):
        if self.video_log_function is not None:
            numpy_array_video = self.video_log_function(self.current_generation, best, config)
            if numpy_array_video is not None:
                wandb.log({"video": wandb.Video(numpy_array_video, fps=15, format="gif")})

    def species_stagnant(self, sid, species):
        pass

    def info(self, msg):
        pass
