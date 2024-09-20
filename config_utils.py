import tempfile
import re


def create_temp_config_file(base_config_file, hyperparams):
    # Read the base config file
    with open(base_config_file, 'r') as f:
        config_data = f.read()

    # Modify the config data with hyperparameters
    for key, value in hyperparams.items():
        config_data = replace_config_value(config_data, key, value)

    # Create a temporary config file
    temp_config = tempfile.NamedTemporaryFile(mode='w', delete=False)
    temp_config.write(config_data)
    temp_config.close()
    return temp_config.name


def replace_config_value(config_data, key, value):
    # Replace the line that starts with the key
    pattern = rf'^(\s*{re.escape(key)}\s*=).*'
    replacement = rf'\1 {value}'
    config_data = re.sub(pattern, replacement, config_data, flags=re.MULTILINE)
    return config_data
