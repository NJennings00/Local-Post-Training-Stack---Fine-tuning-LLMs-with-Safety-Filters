# data/__main__.py
# This provides the required command-line interface to test the downloader.

import argparse
import os
import json
from datasets import load_dataset, load_from_disk # Used to load processed samples
from .utils import load_config
from .downloader import DatasetDownloader
from .preprocessor import Preprocessor

# Define the root path for saving samples
SAMPLES_DIR = 'data/samples/'

def main():
    """
    CLI entry point for the data module. Allows downloading and processing datasets.
    """
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Data Ingestion and Preprocessing Pipeline.")
    # Define the required command line arguments
    parser.add_argument(
        '--download', 
        type=str, 
        nargs='?', 
        const='all', 
        default=None,
        help="Download and save a dataset slice (e.g., '--download sft_train' or '--download all')."
    )
    # Define the process/tokenize command
    parser.add_argument(
        '--process',
        type=str,
        nargs='?',
        const='all',
        default=None,
        help="Load raw data, preprocess, tokenize, and save (e.g., '--process sft_train' or '--process all')."
    )
    # Parse arguments provided by the user
    args = parser.parse_args()
    
    # Check if a command was provided
    if not args.download and not args.process:
        parser.print_help()
        return

    try:
        # Load the central configuration file
        config = load_config()
    except FileNotFoundError:
        print("Error: Config file 'configs/data_config.yaml' not found. Ensure it exists.")
        return
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    # --- Downloader Logic ---
    if args.download:
        downloader = DatasetDownloader(config)
        
        # Determine which datasets to download
        keys_to_download = []
        if args.download == 'all':
            keys_to_download = list(config['data_sources'].keys())
        elif args.download in config['data_sources']:
            keys_to_download = [args.download]
        else:
            print(f"Unknown dataset key '{args.download}'. Available keys: {list(config['data_sources'].keys())}")
            return
            
        # Execute the download for each selected key
        for key in keys_to_download:
            downloader.download_and_save(key)

    # --- Preprocessing Logic ---
    if args.process:
        preprocessor = Preprocessor(config)
        
        # Determine which datasets to process
        keys_to_process = []
        if args.process == 'all':
            keys_to_process = list(config['data_sources'].keys())
        elif args.process in config['data_sources']:
            keys_to_process = [args.process]
        else:
            print(f"Unknown dataset key '{args.process}'. Available keys: {list(config['data_sources'].keys())}")
            return
            
        # Execute the processing for each selected key
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        for key in keys_to_process:
            cfg = config['data_sources'][key]
            raw_path = cfg['save_path']
            
            # 1. Load the raw data slice from disk (as saved by the downloader)
            if not os.path.exists(raw_path):
                print(f"Raw file not found at {raw_path}. Please run '--download {key}' first.")
                continue
                
            try:
                # Load the data from the saved JSONL file
                raw_dataset = load_dataset('json', data_files=raw_path, split='train')
                
                print(f"\n-> Processing raw data from {raw_path}...")
                
                # 2. Process and tokenize the dataset
                processed_ds = preprocessor.process_and_tokenize(raw_dataset, key)
                
                # 3. Save the tokenized dataset to the samples folder
                sample_save_path = os.path.join(SAMPLES_DIR, f"{key}_tokenized")
                processed_ds.save_to_disk(sample_save_path)
                
                print(f"Saved tokenized dataset for {key} to {sample_save_path}")
                # Print a sample to verify the output (e.g., input IDs)
                print("--- Sample Example (Input IDs) ---")
                print(processed_ds[0]['input_ids'][:20]) # Show first 20 tokens
                print("----------------------------------")

            except Exception as e:
                print(f"Error during processing {key}: {e}")


if __name__ == '__main__':
    main()