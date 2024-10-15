import os
import pickle
from wandb_reporter import WandbReporter
from multiprocessing import cpu_count

import numpy as np
import neat
import gymnasium
from config_files_utils import config_to_dict

import torch
from encoding.vae import VAE  # Replace 'vae_model' with the actual module name

device = torch.device("cpu")
torch.set_num_threads(1)

# Global variables to be initialized in each process
global_env = None
global_num_episodes = None
global_vae = None

def gym_initializer(env_spec_id, env_kwargs, num_episodes):
    global global_env
    global global_num_episodes

    global_env = gymnasium.make(env_spec_id, **env_kwargs)
    global_num_episodes = num_episodes


def gym_evaluate_genome(genome, config):
    global global_env
    global global_num_episodes

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = global_env

    total_reward = 0.0
    for _ in range(global_num_episodes):
        observation, _ = env.reset()
        done = False
        while not done:
            if isinstance(env.action_space, gymnasium.spaces.Discrete):
                action = np.argmax(net.activate(observation))
            else:
                action = np.array(net.activate(observation))
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

    return total_reward / global_num_episodes

def cart_initializer(env_spec_id, env_kwargs, num_episodes):
    global global_env
    global global_num_episodes
    global global_vae

    global_env = gymnasium.make(env_spec_id, **env_kwargs)
    global_num_episodes = num_episodes
    # Load VAE model in each process
    vae_model = VAE(32)
    vae_model.load_state_dict(torch.load('encoding/vae.pickle', weights_only=True))
    vae_model.to(device)
    vae_model.eval()
    global_vae = vae_model

def cart_evaluate_genome(genome, config):
    global global_env
    global global_num_episodes
    global global_vae

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = global_env
    vae = global_vae

    total_reward = 0.0
    for _ in range(global_num_episodes):
        observation, _ = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                # Resize and normalize observation to fit VAE input requirements
                observation_resized = observation[20:84, 16:80, :]
                observation_tensor = torch.from_numpy(observation_resized).float().to(device)

                x = vae.preprocess(observation_tensor)
                mu, log_var = vae.encoder(x)
                z = vae.reparameterize(mu, log_var)
                # Detach z, convert to numpy, and flatten for action computation
                z_np = z.detach().cpu().numpy().flatten()

            # Compute action using the latent vector z
            if isinstance(env.action_space, gymnasium.spaces.Discrete):
                action = np.argmax(net.activate(z_np))
            else:
                action = np.array(net.activate(z_np))

            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

    return total_reward / global_num_episodes


class VideoLogFunction:
    def __init__(self, env_spec_id, env_kwargs, visualisation_interval=50):
        self.env_spec_id = env_spec_id
        self.env_kwargs = env_kwargs
        self.visualisation_interval = visualisation_interval

    def __call__(self, current_generation, best_genome, config):
        vae = VAE(32)
        vae.load_state_dict(torch.load('encoding/vae.pickle', map_location=device, weights_only=True))
        vae.to(device)
        vae.eval()


        # Decide whether to log the video based on the generation interval
        if current_generation % self.visualisation_interval == 0:
            # Create the environment with render_mode='rgb_array' for video recording
            env = gymnasium.make(self.env_spec_id, **self.env_kwargs, render_mode='rgb_array')
            net = neat.nn.FeedForwardNetwork.create(best_genome, config)
            observation, _ = env.reset()
            frames = []
            done = False
            step = 0
            while not done:
                with torch.no_grad():
                    observation_resized = observation[20:84, 16:80, :]
                    observation_tensor = torch.from_numpy(observation_resized).float().to(device)
                    x = vae.preprocess(observation_tensor)
                    mu, log_var = vae.encoder(x)
                    z = vae.reparameterize(mu, log_var)
                    z_np = z.detach().numpy().flatten()

                if isinstance(env.action_space, gymnasium.spaces.Discrete):
                    action = np.argmax(net.activate(z_np))
                else:
                    action = np.array(net.activate(z_np))
                observation, _, terminated, truncated, _ = env.step(action)
                frame = env.render()
                frames.append(frame.transpose(2, 0, 1))  # Adjust dimensions for wandb.Video
                done = terminated or truncated
                step += 1

            numpy_array_video = np.array(frames)

            return numpy_array_video
        else:
            return None


def run(config_file: str, env, num_generations=None, checkpoint=None,
        num_tests=5, num_cores=1, wandb_project_name=None, show_species_detail=True, record_video=False):
    print("Charging environment:", env.spec.id)
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config_files", config_file)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    if checkpoint is not None:
        pop = neat.Checkpointer.restore_checkpoint(checkpoint, config)
    else:
        pop = neat.Population(config)

    with open("wandb_api_key.txt", "r") as f:
        wandb_key = f.read().strip()
    config_dict = config_to_dict(config_path)
    if record_video:
        video_log_function = VideoLogFunction(env.spec.id, env.spec.kwargs, visualisation_interval=int(num_generations / 10))
    else:
        video_log_function = None
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
    pop.add_reporter(neat.Checkpointer(
        generation_interval=100,
        filename_prefix="checkpoint-" + env.spec.id + "-"
    ))
    pe = neat.parallel.ParallelEvaluator(
        num_workers=num_cores,
        eval_function=cart_evaluate_genome,
        initializer=cart_initializer,
        initargs=(env.spec.id, env.spec.kwargs, num_tests)
    )

    print("Configuration", pop.config.genome_config)
    gen_best = pop.run(pe.evaluate, num_generations)

    env_args_str = [key for key, value in env.spec.kwargs.items() if value]
    result_path = os.path.join(local_dir, "visualisations", env.spec.id, *env_args_str)
    os.makedirs(result_path, exist_ok=True)
    with open(os.path.join(result_path, 'best_genome.pickle'), 'wb') as f:
        pickle.dump(gen_best, f)

    cart_initializer(env.spec.id, env.spec.kwargs, num_tests)
    score = cart_evaluate_genome(gen_best, config)

    return score


if __name__ == '__main__':
    env_instance = gymnasium.make(
        'CarRacing-v3',
    )

    run(config_file="config-car_racing",
        env=env_instance,
        num_generations=1000,
        num_tests=2,
        num_cores=cpu_count(),
        wandb_project_name="neat-gym",
        record_video=True
        )

    env_instance.close()
