from evolve_mnist import run as run_mnist
from evolve_gym import run as run_gym
from config_files_utils import create_temp_config_file
import os
import sys
import wandb
import gymnasium

if __name__ == '__main__':
    print("CPUS_PER_JOB:", os.environ['CPUS_PER_JOB'])
    print("NUM_GENERATIONS:", os.environ['NUM_GENERATIONS'])
    project = os.environ['WANDB_PROJECT']

    def main_mnist():
        wandb.init(project="neat-mnist")

        # Create a temporary config file with these hyperparameters
        temp_config_file = create_temp_config_file("config_files/config-mnist", wandb.config)

        print("Running MNIST")
        print("Hyperparameters:", wandb.config)

        # Run the NEAT algorithm
        score = run_mnist(
            config_file=temp_config_file,
            num_generations=int(os.environ['NUM_GENERATIONS']),
            num_cores=int(os.environ['CPUS_PER_JOB']),
            subset_size=1000,
            wandb_project_name=project,
            show_species_detail=False
        )

        print("Val score:", score)
        wandb.log({"val_score": score})


    def main_gym():
        wandb.init(project="neat-gym")

        # Create a temporary config file with these hyperparameters
        temp_config_file = create_temp_config_file("config_files/config-ant", wandb.config)

        print("Running ANT")
        print("Hyperparameters:", wandb.config)

        env_instance = gymnasium.make(
            'Ant-v5',
            terminate_when_unhealthy=True,
        )
        # Run the NEAT algorithm
        score = run_gym(
            config_file=temp_config_file,
            env=env_instance,
            penalize_inactivity=False,
            num_generations=int(os.environ['NUM_GENERATIONS']),
            num_tests=2,
            num_cores=int(os.environ['CPUS_PER_JOB']),
            wandb_project_name=project,
            show_species_detail=False,
            record_video=True
        )

        env_instance.close()

        print("Val score:", score)
        wandb.log({"val_score": score})


    with open("wandb_api_key.txt", "r") as f:
        wandb_key = f.read().strip()
    wandb.login(key=wandb_key)

    if project == "neat-mnist":
        wandb.agent(os.environ['SWEEP_ID'], function=main_mnist, project=project, entity="tcazalet_airo", count=1)
    elif project == "neat-gym":
        wandb.agent(os.environ['SWEEP_ID'], function=main_gym, project=project, entity="tcazalet_airo", count=1)
    else:
        raise ValueError("Unknown project")