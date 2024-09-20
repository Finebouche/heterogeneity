import sys
import tempfile
import re
import json
from evolve import run
from config_utils import create_temp_config_file


if __name__ == '__main__':
    # Read hyperparameters from command-line arguments
    if len(sys.argv) < 4:
        print("Usage: python evolve_script.py conn_add_prob conn_delete_prob num_hidden")
        sys.exit(1)

    hyperparams = {
        'conn_add_prob': sys.argv[1],
        'conn_delete_prob': sys.argv[2],
        'num_hidden': sys.argv[3],
        # Add other hyperparameters as needed
    }

    # Convert hyperparameters to appropriate types
    try:
        hyperparams['conn_add_prob'] = float(hyperparams['conn_add_prob'])
        hyperparams['conn_delete_prob'] = float(hyperparams['conn_delete_prob'])
        hyperparams['num_hidden'] = int(hyperparams['num_hidden'])
        # Convert other hyperparameters as needed
    except ValueError as e:
        print(f"Error converting hyperparameters: {e}")
        sys.exit(1)

    # Create a temporary config f    temp_config_file = create_temp_config_file("config-mnist", hyperparams)ile with these hyperparameters

    # Run the NEAT algorithm
    accuracy = run(
        config_file=temp_config_file,
        penalize_inactivity=False,
        num_generations=100,
        num_tests=2,
        num_cores=1,
        subset_size=1000,
        wandb_project_name = "neat-mnist"
    )
    results = {"accuracy": accuracy}
    print(json.dumps(results))
