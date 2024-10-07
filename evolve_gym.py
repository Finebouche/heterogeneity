import os
import pickle
from wandb_reporter import WandbReporter
from multiprocessing import cpu_count

import numpy as np
import neat
import gymnasium
from config_files_utils import config_to_dict

# Global variables to be initialized in each process
global_env = None
global_penalize_inactivity = None
global_num_episodes = None


def gym_initializer(env_spec_id, env_kwargs, penalize_inactivity, num_episodes):
    global global_env
    global global_penalize_inactivity
    global global_num_episodes

    global_env = gymnasium.make(env_spec_id, **env_kwargs)
    global_penalize_inactivity = penalize_inactivity
    global_num_episodes = num_episodes


def gym_evaluate_genome(genome, config):
    global global_env
    global global_penalize_inactivity
    global global_num_episodes

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = global_env

    def compute_action_discrete(net, observation):
        activation = net.activate(observation)
        action = np.argmax(activation)
        return action, activation[action]

    def compute_action_box(net, observation):
        action = net.activate(observation)
        norm = np.linalg.norm(action)
        return action, norm

    if isinstance(env.action_space, gymnasium.spaces.Discrete):
        compute_action = compute_action_discrete
    else:
        compute_action = compute_action_box

    total_reward = 0.0
    for _ in range(global_num_episodes):
        observation, _ = env.reset()
        done = False
        while not done:
            action, norm = compute_action(net, observation)
            if (observation[2] < 0.01 or norm < 1.8) and global_penalize_inactivity:
                total_reward -= 1
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
                if isinstance(env.action_space, gymnasium.spaces.Discrete):
                    action = np.argmax(net.activate(observation))
                else:
                    action = net.activate(observation)
                observation, _, terminated, truncated, _ = env.step(action)
                frame = env.render()
                frames.append(frame.transpose(2, 0, 1))  # Adjust dimensions for wandb.Video
                done = terminated or truncated
                step += 1
            print(f"Generation {current_generation}: Video recorded with {step} steps")

            numpy_array_video = np.array(frames)

            return numpy_array_video
        else:
            return None


def run(config_file: str, env, penalize_inactivity=False, num_generations=None, checkpoint=None,
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
    visualisation_interval = int(num_generations / 10)
    if record_video:
        video_log_function = VideoLogFunction(env.spec.id, env.spec.kwargs, visualisation_interval)
    else:
        video_log_function = None
    config_dict = config_to_dict(config_file)
    wandb_reporter = WandbReporter(
        project_name=wandb_project_name,
        config=config_dict,
        tags=["neat", env.spec.id],
        api_key=wandb_key,
        video_log_function=video_log_function
    )
    pop.add_reporter(wandb_reporter)
    pop.add_reporter(neat.StdOutReporter(show_species_detail))
    pop.add_reporter(neat.Checkpointer(
        generation_interval=int(num_generations / 10),
        time_interval_seconds=1800,
        filename_prefix="checkpoint-" + env.spec.id + "-"
    ))

    pe = neat.parallel.ParallelEvaluator(
        num_workers=num_cores,
        eval_function=gym_evaluate_genome,
        initializer=gym_initializer,
        initargs=(env.spec.id, env.spec.kwargs, penalize_inactivity, num_tests)
    )

    print("Configuration", pop.config.genome_config)
    gen_best = pop.run(pe.evaluate, num_generations)

    env_args_str = [key for key, value in env.spec.kwargs.items() if value]
    result_path = os.path.join(local_dir, "visualisations", env.spec.id, *env_args_str)
    os.makedirs(result_path, exist_ok=True)
    with open(os.path.join(result_path, 'best_genome.pickle'), 'wb') as f:
        pickle.dump(gen_best, f)

    env.close()


if __name__ == '__main__':
    env_instance = gymnasium.make(
        'Ant-v5',
        terminate_when_unhealthy=True,
    )

    run(config_file="config-ant",
        env=env_instance,
        penalize_inactivity=False,
        num_generations=500,
        num_tests=2,
        num_cores=cpu_count(),
        wandb_project_name="neat-gym",
        record_video=True
        )
