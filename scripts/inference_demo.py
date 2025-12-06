# Import the pipeline function and set_seed for reproducible results.
from transformers import pipeline, set_seed

# --------------------------------------------------------------------------------
# Function: run_inference_demo
# Purpose: Initializes the text generation pipeline using the DistilGPT2 model
#          and runs a test prompt to demonstrate baseline functionality.
# Inputs:
#   - model_name (str): The identifier of the model to be used.
#   - prompt (str): The initial text sequence for the model to continue.
# Outputs:
#   - None (The function prints the generated text to the console).
# --------------------------------------------------------------------------------
def run_inference_demo(model_name: str, prompt: str):
    # Set a random seed to ensure the generated text is the same across runs.
    # This is critical for reproducibility.
    set_seed(42)
    
    # Initialize the text generation pipeline.
    # The 'pipeline' abstraction handles loading the model, tokenizer, and placing it
    # on the best available device (GPU if CUDA is set up, otherwise CPU).
    print(f"Initializing text generation pipeline with model: {model_name}...")
    generator = pipeline(
        "text-generation", # Specify the task
        model=model_name   # Specify the model to use (will download if not local)
    )

    # Define generation parameters.
    # max_length: limits the total length of the input + generated text.
    # num_return_sequences: how many unique outputs to generate.
    # pad_token_id: for DistilGPT2, the EOS token ID is used for padding.
    print(f"Running inference with prompt: '{prompt}'")
    generated_output = generator(
        prompt,
        max_length=50,
        num_return_sequences=1,
        truncation=True,
        pad_token_id=generator.tokenizer.eos_token_id
    )

    # --- Print Results ---
    print("\n" + "="*50)
    print("✨ GENERATED BASELINE OUTPUT ✨")
    # The output is a list of dictionaries; extract the generated text.
    print(generated_output[0]['generated_text'])
    print("="*50 + "\n")


# --- Main Execution Block ---
if __name__ == "__main__":
    # The name of the model to use (will fetch from Hugging Face Hub).
    MODEL_ID = "distilgpt2"
    # The test prompt for the model to complete.
    TEST_PROMPT = "The capital of France is"
    
    # Run the demo function.
    run_inference_demo(MODEL_ID, TEST_PROMPT)