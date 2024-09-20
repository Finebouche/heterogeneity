import tempfile
import re


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

