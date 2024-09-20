import os
import sys
import tempfile
import re
import json
from evolve import run

# Include your NEAT code here, including the 'run' function and any other necessary functions

def create_temp_config_file(base_config_file, hyperparams):
    with open(base_config_file, 'r') as f:
        config_data = f.read()

    for key, value in hyperparams.items():
        config_data = replace_config_value(config_data, key, value)

    temp_config = tempfile.NamedTemporaryFile(mode='w', delete=False)
    temp_config.write(config_data)
    temp_config.close()
    return temp_config.name


def replace_config_value(config_data, key, value):
    pattern = rf'^(\s*{re.escape(key)}\s*=).*'
    replacement = rf'\1 {value}'
    config_data = re.sub(pattern, replacement, config_data, flags=re.MULTILINE)
    return config_data


def main():
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


    # Create a temporary config file with these hyperparameters
    temp_config_file = create_temp_config_file("config-mnist", hyperparams)

    # Run the NEAT algorithm
    accuracy = run(
        config_file=temp_config_file,
        penalize_inactivity=False,
        num_generations=100,
        num_tests=2,
        num_cores=1,
        subset_size=1000
    )

    # Save the result to a file that can be retrieved
    result = {'accuracy': accuracy}
    with open('result.json', 'w') as f:
        json.dump(result, f)

    # Clean up the temporary config file
    os.remove(temp_config_file)


if __name__ == '__main__':
    main()
