# train/train_cli.py
# A simple command-line interface (CLI) that uses the implemented SFTTrainer class to kick off the job based on the chosen config file.

import argparse                           # Import argparse for command-line argument parsing.
from sft_trainer import SFTTrainer        # Import the SFTTrainer class we just created.
import torch.cuda                         # Import CUDA functions for memory management.
import os                                 # Import OS for creating necessary directories.

# Define the main function for the CLI
def main():
    """
    Main function to parse arguments and start the training job.
    """
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description="Run Supervised Fine-Tuning (SFT) using LoRA and resource-aware configuration."
    )
    # Add the required config path argument
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the training config YAML file (e.g., configs/local_gpu_config.yaml)."
    )
    # Add an optional flag for resuming training
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Flag to resume training from the latest checkpoint in the output directory."
    )
    
    # Parse the command line arguments
    args = parser.parse_args()

    # Clear CUDA cache before starting, if running on GPU, to prevent fragmentation OOM errors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        # Create the checkpoints directory if it doesn't exist
        os.makedirs("checkpoints", exist_ok=True)
        # Initialize the SFTTrainer with the specified configuration
        trainer = SFTTrainer(args.config)
        # Start the training process
        trainer.train(resume_from_checkpoint=args.resume)
        
    except FileNotFoundError as e:
        # Handle cases where config or dataset file is missing
        print(f"Error: A required file was not found: {e}")
        print("Please ensure the data/tokenized/alpaca_processed dataset exists and the config path is correct.")
    except Exception as e:
        # Catch any other unexpected error during training
        print(f"An unexpected error occurred during training: {e}")
        
# Check if the script is executed directly
if __name__ == "__main__":
    main()