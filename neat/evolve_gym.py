import os
import pickle
from multiprocessing import cpu_count

import numpy as np
import neat
import gymnasium
import torch

from neat.config_files.config_files_utils import config_to_dict
from encoding.vae import load_vae_model
from wandb_reporter import WandbReporter

device = torch.device("cpu")
torch.set_num_threads(1)

# Global variables to be initialized in each process
global_env = None
global_num_episodes = None
global_vae = None
global_unique_value = None
global_random_values = None

def process_observation_with_vae(observation, vae, device):
    """Process the observation using the VAE."""
    with torch.no_grad():
        # Resize and normalize observation to fit VAE input requirements
        observation_resized = observation[20:84, 16:80, :]
        observation_tensor = torch.from_numpy(observation_resized).float().to(device)

        x = vae.preprocess(observation_tensor)
        mu, log_var = vae.encoder(x)
        z = vae.reparameterize(mu, log_var)
        input_data = z.cpu().numpy().flatten()

    return input_data


def gym_initializer(env_spec_id, env_kwargs, num_episodes, unique_value=None, random_values=None, vae_path=None):
    """Initializer function for parallel evaluation."""
    global global_env, global_num_episodes, global_vae
    global global_unique_value, global_random_values

    global_env = gymnasium.make(env_spec_id, **env_kwargs)
    global_num_episodes = num_episodes
    global_unique_value = unique_value
    global_random_values = random_values

    # Load VAE model if present
    if vae_path is not None:
        global_vae = load_vae_model(vae_path, device)
    else:
        global_vae = None


def gym_evaluate_genome(genome, config):
    """Evaluate a single genome."""
    global global_env, global_num_episodes, global_vae
    global global_unique_value, global_random_values

    env = global_env
    vae = global_vae
    unique_value = global_unique_value
    random_values = global_random_values

    net = neat.nn.FeedForwardNetwork.create(genome, config, unique_value, random_values)

    total_reward = 0.0
    for _ in range(global_num_episodes):
        observation, _ = env.reset()
        done = False
        while not done:
            if global_vae is not None:
                input_data = process_observation_with_vae(observation, vae, device)
            else:
                input_data = observation


            if isinstance(env.action_space, gymnasium.spaces.Discrete):
                action = np.argmax(net.activate(input_data))
            else:
                action = np.array(net.activate(input_data))

            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

    return total_reward / global_num_episodes


class VideoLogFunction:
    def __init__(self, env_spec_id, env_kwargs, unique_value, random_values, vae_path=None, visualisation_interval=50):
        self.env_spec_id = env_spec_id
        self.env_kwargs = env_kwargs
        self.visualisation_interval = visualisation_interval
        self.unique_value = unique_value
        self.random_values = random_values
        self.vae = load_vae_model(vae_path, device) if vae_path else None

    def __call__(self, current_generation, best_genome, config):
        if current_generation % self.visualisation_interval != 0:
            return None

        # Create the environment with render_mode='rgb_array' for video recording
        env = gymnasium.make(self.env_spec_id, **self.env_kwargs, render_mode='rgb_array')
        net = neat.nn.FeedForwardNetwork.create(best_genome, config, self.unique_value, self.random_values)
        observation, _ = env.reset()
        frames = []
        done = False

        while not done:
            if self.vae is not None:
                input_data = process_observation_with_vae(observation, self.vae, device)
            else:
                input_data = observation

            if isinstance(env.action_space, gymnasium.spaces.Discrete):
                action = np.argmax(net.activate(input_data))
            else:
                action = np.array(net.activate(input_data))

            observation, _, terminated, truncated, _ = env.step(action)
            frame = env.render()
            frames.append(frame.transpose(2, 0, 1))  # dimensions order for wandb.Video
            done = terminated or truncated

        env.close()
        return np.array(frames)


def run(config_file: str, env, num_generations=None, checkpoint=None,
        num_tests=5, num_cores=1, wandb_project_name=None, show_species_detail=True, record_video=False,
        unique_value=None, random_values=None, vae_path=None):
    print("Charging environment:", env.spec.id)
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config_files", config_file)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Checkpointer.restore_checkpoint(checkpoint, config) if checkpoint else neat.Population(config)

    with open("../wandb_api_key.txt", "r") as f:
        wandb_key = f.read().strip()
    config_dict = config_to_dict(config_path)

    if record_video:
        video_log_function = VideoLogFunction(
            env.spec.id,
            env.spec.kwargs,
            unique_value,
            random_values,
            vae_path=vae_path,
            visualisation_interval=int(num_generations / 10))
    else:
        video_log_function = None


    # Add unique_value and random_values to the config_dict to log them
    config_dict['unique_value'] = unique_value
    config_dict['random_values'] = random_values
    wandb_reporter = WandbReporter(
        project_name=wandb_project_name,
        tags=[env.spec.id],
        api_key=wandb_key,
        log_config=config_dict,
        log_network=True,
        video_log_function=video_log_function
    )
    pop.add_reporter(wandb_reporter)
    pop.add_reporter(neat.StdOutReporter(show_species_detail))
    # pop.add_reporter(neat.Checkpointer(
    #     generation_interval=100,
    #     filename_prefix=f"checkpoint-{env.spec.id}-"
    # ))
    pe = neat.parallel.ParallelEvaluator(
        num_workers=num_cores,
        eval_function=gym_evaluate_genome,
        initializer=gym_initializer,
        initargs=(env.spec.id, env.spec.kwargs, num_tests, unique_value, random_values, vae_path)
    )

    best_genome = pop.run(pe.evaluate, num_generations)

    # Save best model at the end
    env_args_str = [key for key, value in env.spec.kwargs.items() if value]
    result_path = os.path.join(local_dir, "visualisations", env.spec.id, *env_args_str)
    os.makedirs(result_path, exist_ok=True)
    with open(os.path.join(result_path, 'best_genome.pickle'), 'wb') as f:
        pickle.dump(best_genome, f)


    # Evaluate the best genome
    gym_initializer(env.spec.id, env.spec.kwargs, num_tests, unique_value, random_values, vae_path)
    score = gym_evaluate_genome(best_genome, config)
    return score

if __name__ == '__main__':
    env_instance = gymnasium.make('CarRacing-v3')

    run(config_file="config-car_racing",
        env=env_instance,
        num_generations=4000,
        num_tests=1,
        num_cores=cpu_count(),
        wandb_project_name="neat-het",
        record_video=True,
        unique_value=None,
        random_values=None,
        vae_path ='../encoding/vae_cpu.pickle'  # Specify the path to your VAE model
        )

    env_instance.close()
