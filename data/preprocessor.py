# data/preprocessor.py
# This file contains the logic for data cleaning, integrity checks, formatting, 
# and the tokenizer wrapper for memory control.

from transformers import AutoTokenizer          # Tool for loading the LLM tokenizer
from datasets import Dataset                    # Hugging Face datasets library
from typing import Dict, Any, List, Optional
import random

class TokenizerWrapper:
    """
    Wraps the Hugging Face tokenizer to handle tokenization parameters
    including padding, truncation, and deterministic shuffling.
    """
    
    def __init__(self, model_name: str, max_length: int, seed: int):
        """
        Initializes the wrapper by loading the model's tokenizer and setting parameters.
        
        Inputs:
          model_name (str): The name of the model (e.g., 'distilgpt2').
          max_length (int): The maximum sequence length for truncation and padding.
          seed (int): The random seed for deterministic operations.
        
        Outputs:
          None
        """
        # Load the pre-trained tokenizer for the specified model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set the tokenizer's padding token to the EOS token if not defined
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Store key tokenization parameters
        self.max_length = max_length
        self.seed = seed

    def tokenize_function(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        A function to apply to the dataset using .map() for tokenization.

        Inputs:
          examples (dict): A dictionary of text examples (batch) from the dataset.
        
        Outputs:
          dict: A dictionary containing 'input_ids' and 'attention_mask'.
        """
        # Call the tokenizer on the input text batch
        return self.tokenizer(
            examples['text'],
            truncation=True,            # Truncate sequences longer than max_length
            padding='max_length',       # Pad sequences to max_length
            max_length=self.max_length, # The specified maximum length (128)
            return_attention_mask=True  # Include the attention mask
        )

    def shuffle_dataset(self, dataset: Dataset) -> Dataset:
        """
        Shuffles the dataset deterministically using the configured seed.
        
        Inputs:
          dataset (Dataset): The Hugging Face Dataset object.
        
        Outputs:
          Dataset: The shuffled Dataset object.
        """
        # Apply deterministic shuffling using the class seed
        return dataset.shuffle(seed=self.seed)


class Preprocessor:
    """
    A class to clean, format, and tokenize raw datasets into a model-ready state.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Preprocessor with configuration and the TokenizerWrapper.
        
        Inputs:
          config (dict): The loaded configuration dictionary.
        
        Outputs:
          None
        """
        # Store configuration for later use
        self.config = config

        # Get configuration dictionaries from the top-level config
        model_cfg = config['model_config']
        preproc_cfg = config['preproc_config']

        # Initialize the TokenizerWrapper with model parameters
        self.tokenizer_wrapper = TokenizerWrapper(
            model_name=model_cfg['model_name'],
            max_length=preproc_cfg['max_seq_length'],
            seed=preproc_cfg['seed']
        )
        # Store the template for SFT data
        self.alpaca_template = preproc_cfg['alpaca_template']
        # Set the global random seed for any non-Hugging Face random operations
        random.seed(preproc_cfg['seed'])

    def format_alpaca_example(self, example: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """
        Formats an instruction-following example into a single text string.
        Returns {'text': None} if the instruction or output is malformed.
        """
        instruction = example.get('instruction')
        output = example.get('output')

        # Check for malformed inputs (None or effectively empty string).
        # We check that instruction AND the stripped output are truthy (non-None, non-empty).
        if not instruction or not str(output).strip():
            return {'text': None} # Returning None will make check_integrity filter it out

        # Use the stored template
        text = self.alpaca_template.format(
            instruction=instruction,
            # Ensure 'input' field is handled gracefully if not provided
            input=example.get('input', ''), 
            output=output
        )
        return {'text': text}

    def check_integrity(self, example: Dict[str, Any]) -> bool:
        """
        Performs data integrity checks on an example, ensuring no nulls or empty strings.
        
        Inputs:
          example (dict): A single dictionary entry from the processed dataset.
        
        Outputs:
          bool: True if the example is valid, False otherwise (for filtering).
        """
        # Check for null or empty string in the final formatted text field
        text = example.get('text', None)
        # Ensure the text exists and has a non-zero length after stripping whitespace
        if text is None or not str(text).strip():
            # This handles the 'no nulls' and 'not empty' acceptance criteria
            return False
            
        # TODO: Check if the tokenized length is too short/long (can be done later after tokenization)
        # For now, rely on `truncation=True` in the tokenizer wrapper
        return True

    def process_and_tokenize(self, dataset: Dataset, dataset_type: str) -> Dataset:
        """
        Applies cleaning, formatting, integrity checks, and tokenization to a dataset.

        Inputs:
        dataset (Dataset): The raw Hugging Face Dataset object.
        dataset_type (str): The type of dataset (e.g., 'sft_train').
        
        Outputs:
        Dataset: The fully processed and tokenized Dataset object.
        """
        
        # 1. Format the raw data into a single 'text' string and rename columns
        
        # --- CASE 1: SFT TRAINING DATA (Alpaca) ---
        if dataset_type == 'sft_train':
            # Map the formatting function over the entire dataset, creating the 'text' column
            dataset = dataset.map(
                self.format_alpaca_example, 
                # Remove original columns as they are replaced by the formatted 'text'
                remove_columns=[col for col in dataset.column_names if col not in ['text']],
                desc="Formatting SFT data"
            )
        
        # --- CASE 2: ADVERSARIAL EVAL DATA (JailbreakBench) ---
        elif dataset_type == 'adversarial_eval':
            # JailbreakBench uses the column name 'jailbreak_prompt' for the prompt
            if 'Behavior' in dataset.column_names:
                # Rename the prompt column to the expected 'text' column
                dataset = dataset.rename_column('Behavior', 'text')
            else:
                 # Add a warning in case the column name changes in the future
                 print(f"Warning: Adversarial dataset does not contain 'Behavior' column. Check raw data.")
            
        # --- CASE 3: HELPFULNESS EVAL DATA (SQuAD) ---
        elif dataset_type == 'helpfulness_eval':
            # SQuAD uses 'question' for the simple prompt we want to tokenize for quick eval.
            if 'question' in dataset.column_names:
                dataset = dataset.rename_column('question', 'text')

        # --- Remaining steps are universal after 'text' column is ensured ---
        
        # 2. Integrity check: Filter out bad/empty examples
        dataset = dataset.filter(
            self.check_integrity, 
            desc="Filtering for data integrity"
        )

        # 3. Tokenize the formatted text
        # The tokenizer map creates 'input_ids' and 'attention_mask'.
        tokenized_dataset = dataset.map(
            self.tokenizer_wrapper.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names, 
            desc="Tokenizing dataset"
        )
        
        print(f"Preprocessing and tokenization complete. New features: {tokenized_dataset.column_names}")
        return tokenized_dataset