from evolve_mnist import run as run_mnist
from evolve_gym import run as run_gym
from config_files_utils import create_temp_config_file
import os
import sys
import wandb
import gymnasium

if __name__ == '__main__':
    # Read hyperparameters from command-line arguments
    try:
        hyperparams = {
            'conn_add_prob': float(os.environ['CONN_ADD_PROB']),
            'conn_delete_prob': float(os.environ['CONN_DELETE_PROB']),
            'num_hidden': int(os.environ['NUM_HIDDEN']),
            'activation_options': os.environ['ACTIVATION_OPTIONS'],
            'activation_mutate_rate': float(os.environ['ACTIVATION_MUTATE_RATE']),
            'weight_mutate_rate': float(os.environ['WEIGHT_MUTATE_RATE']),
            'enabled_mutate_rate': float(os.environ['ENABLED_MUTATE_RATE']),
            # Add other hyperparameters as needed
        }
    except KeyError as e:
        print(f"Environment variable {e} not found.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error converting hyperparameters: {e}")
        sys.exit(1)


    def main_mnist():
        # Create a temporary config file with these hyperparameters
        temp_config_file = create_temp_config_file("config_files/config-mnist", hyperparams)

        # Run the NEAT algorithm
        score = run_mnist(
            config_file=temp_config_file,
            num_generations=300,
            num_cores=int(os.environ['CPUS_PER_JOB']),
            subset_size=1000,
            wandb_project_name="neat-mnist"
        )
        wandb.log({"score": score})


    def main_gym():
        # Create a temporary config file with these hyperparameters
        temp_config_file = create_temp_config_file("config_files/config-ant", hyperparams)
        env_instance = gymnasium.make(
            'Ant-v5',
            terminate_when_unhealthy=False,
        )
        # Run the NEAT algorithm
        run_gym(
            config_file=temp_config_file,
            penalize_inactivity=False,
            num_generations=100,
            num_tests=2,
            num_cores=int(os.environ['CPUS_PER_JOB']),
            wandb_project_name="neat-gym"
        )


    with open("wandb_api_key.txt", "r") as f:
        wandb_key = f.read().strip()
    wandb.login(key=wandb_key)

    wandb.agent(os.environ['SWEEP_ID'], function=main_mnist, project="neat-mnist", entity="tcazalet_airo")
