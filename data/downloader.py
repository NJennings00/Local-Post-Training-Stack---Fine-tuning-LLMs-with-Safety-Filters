# data/downloader.py
# This class handles fetching, slicing, and saving datasets, incorporating error handling and deterministic slicing.

import os
import json
from datasets import load_dataset, Dataset # Hugging Face datasets library
from typing import Dict, Any

class DatasetDownloader:
    """
    A class to download, slice, and save small subsets of Hugging Face datasets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Downloader with the project configuration.
        
        Inputs:
          config (dict): The loaded configuration dictionary from data_config.yaml.
        
        Outputs:
          None
        """
        # Store the data source configuration
        self.data_sources = config.get('data_sources', {})
        # Store the global seed for deterministic slicing
        self.seed = config['preproc_config']['seed']
        
    def download_and_save(self, dataset_key: str):
        """
        Downloads a slice of the specified dataset, shuffles it, and saves it locally.

        Inputs:
          dataset_key (str): Key from the 'data_sources' config (e.g., 'sft_train').
        
        Outputs:
          Dataset: The sliced Hugging Face Dataset object.
        """
        # Retrieve the specific configuration for the requested dataset key
        cfg = self.data_sources.get(dataset_key)

        # Check if the configuration key is valid
        if not cfg:
            print(f"Error: Dataset key '{dataset_key}' not found in configuration.")
            return None

        # Extract required parameters from the configuration
        hf_name = cfg['hf_name']         # Hugging Face dataset name
        split = cfg['split']             # Dataset split (e.g., 'train')
        subset_size = cfg['subset_size'] # Target size of the local subset
        save_path = cfg['save_path']     # Local path to save the file
        
        # Safely extract the optional 'config_name' (used by JailbreakBench)
        config_name = cfg.get('config_name') # Returns None if key is missing

        print(f"-> Downloading and processing {hf_name}/{split}...")

        try:
            # Create a dictionary of arguments for the load_dataset function
            load_args = {
                'path': hf_name, # The dataset identifier (the first argument)
                'split': split   # The dataset split (a keyword argument)
            }
            
            # Conditionally add the 'name' argument if 'config_name' was provided in the YAML
            # The 'datasets' library uses 'name' as the argument for config_name
            if config_name:
                load_args['name'] = config_name
            
            # Load the full dataset split from the Hugging Face Hub using dictionary unpacking
            # This ensures only the necessary arguments are passed.
            dataset = load_dataset(load_args['path'], name=load_args.get('name'), split=load_args['split'])
            
            # Use train_test_split to get a deterministic shuffle and slice
            dataset = dataset.shuffle(seed=self.seed)
            
            # Select the first N examples to create the required small slice
            # Ensure the slice size does not exceed the available data
            slice_size = min(subset_size, len(dataset))
            sliced_dataset = dataset.select(range(slice_size))

            # Create the necessary directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save the sliced dataset locally as JSON lines (JSONL)
            sliced_dataset.to_json(save_path, orient='records', lines=True)

            print(f"Saved {slice_size} rows of {hf_name} to {save_path}")
            
            return sliced_dataset

        # Catch exceptions related to network, missing data, or disk issues
        except Exception as e:
            print(f"Error downloading {hf_name}: {e}")
            return None