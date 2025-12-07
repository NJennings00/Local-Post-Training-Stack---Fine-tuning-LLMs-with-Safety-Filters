# scripts/prep_data.py

# Import necessary standard Python libraries
import os
import argparse
import yaml
import logging

import sys
# Calculate the path to the project root directory (one level up from 'scripts')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
# Add the project root to the system path to enable imports like 'from data.downloader...'
sys.path.append(PROJECT_ROOT)

# Import the existing, modular classes from the project structure
from data.downloader import DatasetDownloader
from data.preprocessor import Preprocessor
from datasets import load_dataset, DatasetDict

# Configure basic logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """
    Purpose: Loads configuration settings from a specified YAML file.
    
    Inputs:
      config_path (str): The file path to the YAML configuration file.

    Outputs:
      dict: A dictionary containing the configuration parameters.
    """
    # Open and safely load the YAML configuration file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """
    Purpose: Main orchestration function for the data ingestion and preprocessing pipeline.

    Inputs:
      None (reads arguments from the command line).

    Outputs:
      None (saves the tokenized dataset to disk).
    """
    # 1. Setup and Configuration Loading
    parser = argparse.ArgumentParser(description="Prepare and tokenize instruction-tuning dataset.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info("Configuration loaded successfully.")

    # 2. Data Downloading and Loading
    # Initialize the Downloader with the full configuration
    downloader = DatasetDownloader(config)
    
    # Download and load the raw SFT training dataset. 
    # This step uses the downloader to slice the data, save a JSONL copy, and return a HF Dataset object.
    dataset_key = 'sft_train'
    raw_dataset = downloader.download_and_save(dataset_key)
    
    if raw_dataset is None:
        logger.error(f"Failed to download/load raw dataset for key '{dataset_key}'. Aborting.")
        return

    # 3. Data Preprocessing and Tokenization
    # Initialize the Preprocessor with the full configuration
    preprocessor = Preprocessor(config)
    
    # Process the raw dataset: format the text, run integrity checks, and tokenize
    dataset_type = config['data_sources'][dataset_key]['dataset_type'] # e.g., 'sft_train'
    tokenized_dataset = preprocessor.process_and_tokenize(raw_dataset, dataset_type)
    
    # 4. Splitting and Saving
    
    # Get the test size from the preprocessor configuration
    test_size = config['preproc_config']['test_size']
    seed = config['preproc_config']['seed']
    
    logger.info(f"Splitting tokenized dataset into train/test with test_size={test_size}...")
    # Use the Hugging Face utility to split the dataset
    train_test_split = tokenized_dataset.train_test_split(
        test_size=test_size, 
        seed=seed
    )

    # Get the output path for the *tokenized* data from the configuration
    output_dir = config['paths']['tokenized_output_dir']
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created tokenized output directory: {output_dir}")

    # Save the ENTIRE DatasetDict object to the top-level path.
    # This automatically creates the 'train' and 'test' subfolders along with the 
    # required top-level metadata files, making it a valid DatasetDict directory.
    logger.info(f"Saving combined DatasetDict to {output_dir}...")
    train_test_split.save_to_disk(output_dir)

    logger.info("Data preparation pipeline complete.")

if __name__ == "__main__":
    main()