import os
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
    # Read hyperparameters from environment variables
    hyperparams = {
        'conn_add_prob': os.environ.get('CONN_ADD_PROB'),
        'conn_delete_prob': os.environ.get('CONN_DELETE_PROB'),
        'num_hidden': os.environ.get('NUM_HIDDEN'),
        # Add other hyperparameters as needed
    }

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
