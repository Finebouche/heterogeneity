import tempfile
import re
import configparser


def config_to_dict(file_name):
    # Read the configuration file and convert it to a dictionary
    config = configparser.ConfigParser()
    config.read(file_name)

    # Create an empty dictionary
    config_dict = {}

    # Iterate over all sections and their corresponding key-value pairs
    for section in config.sections():
        config_dict[section] = {}
        for key, value in config.items(section):
            # Try to convert to the appropriate data type (int, float, bool)
            config_dict[section][key] = convert_value(value)

    return config_dict


def convert_value(value):
    """Convert string values to their correct type: int, float, bool, or leave as string."""
    try:
        # Try to convert to integer
        return int(value)
    except ValueError:
        try:
            # Try to convert to float
            return float(value)
        except ValueError:
            # Convert string representations of booleans
            if value.lower() in ['true', 'false']:
                return value.lower() == 'true'
            # Otherwise return as string
            return value


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
