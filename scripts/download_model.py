# Import the necessary components from the Hugging Face transformers library.
# AutoTokenizer loads the vocabulary and pre-processing rules.
# AutoModelForCausalLM loads the model weights for language generation.
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# --- Constants ---
# The name of the small LLM to be used for the project.
MODEL_NAME = "distilgpt2"
# The path where the model and tokenizer will be saved locally.
OUTPUT_DIR = "models/base_llm"

# --------------------------------------------------------------------------------
# Function: download_model_and_tokenizer
# Purpose: Downloads the specified model and its tokenizer from the Hugging Face Hub
#          and saves them to a local directory for offline use.
# Inputs:
#   - model_name (str): The identifier of the pre-trained model on the Hub.
#   - output_dir (str): The local path where the files will be saved.
# Outputs:
#   - None (The function performs I/O by saving files).
# --------------------------------------------------------------------------------
def download_model_and_tokenizer(model_name: str, output_dir: str):
    # Ensure the output directory exists.
    if not os.path.exists(output_dir):
        # Create the directory, including any necessary parent directories.
        os.makedirs(output_dir)
        
    # --- 1. Load and Save the Tokenizer ---
    print(f"Downloading tokenizer for {model_name}...")
    # AutoTokenizer intelligently loads the correct tokenizer class for the model name.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Save the tokenizer files (vocabulary, special tokens, config) to the local disk.
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved successfully to {output_dir}")

    # --- 2. Load and Save the Model ---
    print(f"Downloading model weights for {model_name}...")
    # AutoModelForCausalLM loads the model architecture and pre-trained weights.
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # Save the model's weights and configuration to the local disk.
    model.save_pretrained(output_dir)
    print(f"Model saved successfully to {output_dir}")
    
# --- Main Execution Block ---
if __name__ == "__main__":
    # Call the main function with the defined constants.
    download_model_and_tokenizer(MODEL_NAME, OUTPUT_DIR)
    # Signal completion to the user.
    print("Baseline model download complete. Ready for inference demo.")