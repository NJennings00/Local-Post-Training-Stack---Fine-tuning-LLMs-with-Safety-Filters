# data/utils.py
# This small utility handles loading the YAML configuration.

import yaml 

def load_config(config_path: str = 'configs/data_config.yaml') -> dict:
    """
    Loads the configuration from a specified YAML file.

    Inputs:
      config_path (str): The file path to the YAML configuration.
    
    Outputs:
      dict: The loaded configuration dictionary.
    """
    # Open the configuration file in read mode
    with open(config_path, 'r') as f:
        # Load the YAML content and return it
        config = yaml.safe_load(f)
    return config