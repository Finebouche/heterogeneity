import sys
import json
from evolve import run
from config_utils import create_temp_config_file


if __name__ == '__main__':
    # Read hyperparameters from command-line arguments
    if len(sys.argv) < 5:
        print("Usage: python evolve_script.py conn_add_prob conn_delete_prob num_hidden activation_options")
        sys.exit(1)

    # Convert hyperparameters to appropriate types
    try:
        hyperparams = {
            'conn_add_prob': float(sys.argv[1]),
            'conn_delete_prob': float(sys.argv[2]),
            'num_hidden': int(sys.argv[3]),
            'activation_options': str(sys.argv[4])
            # Add other hyperparameters as needed
        }
    except ValueError as e:
        print(f"Error converting hyperparameters: {e}")
        sys.exit(1)

    # Create a temporary config file with these hyperparameters
    temp_config_file = create_temp_config_file("config-mnist", hyperparams)

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
