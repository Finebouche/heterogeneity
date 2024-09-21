from evolve import run
from config_utils import create_temp_config_file
import os
import sys
import wandb

if __name__ == '__main__':
    # Read hyperparameters from command-line arguments
    try:
        hyperparams = {
            'conn_add_prob': float(os.environ['CONN_ADD_PROB']),
            'conn_delete_prob': float(os.environ['CONN_DELETE_PROB']),
            'num_hidden': int(os.environ['NUM_HIDDEN']),
            'activation_options': os.environ['ACTIVATION_OPTIONS']
            # Add other hyperparameters as needed
        }
    except KeyError as e:
        print(f"Environment variable {e} not found.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error converting hyperparameters: {e}")
        sys.exit(1)


    def main():
        # Create a temporary config file with these hyperparameters
        temp_config_file = create_temp_config_file("config-mnist", hyperparams)

        # Run the NEAT algorithm
        score = run(
            config_file=temp_config_file,
            penalize_inactivity=False,
            num_generations=100,
            num_tests=2,
            num_cores=1,
            subset_size=1000,
            wandb_project_name = "neat-mnist"
        )
        wandb.log({"score": score})

    with open("wandb_api_key.txt", "r") as f:
        wandb_key = f.read().strip()
    wandb.login(key=wandb_key)

    wandb.agent(os.environ['SWEEP_ID'], function=main, project="neat-mnist", entity="tcazalet_airo")
