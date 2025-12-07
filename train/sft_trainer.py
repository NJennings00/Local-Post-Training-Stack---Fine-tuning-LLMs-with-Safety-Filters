# train/sft_trainer.py

import torch                                     # Import PyTorch, the main tensor library.
import yaml                                      # Import YAML for configuration file parsing.
import os                                        # Import OS for path and file operations.
from transformers import (                       # Import core classes from Hugging Face Transformers.
    AutoModelForCausalLM,                        # Class to load any Causal Language Model.
    AutoTokenizer,                               # Class to load any Tokenizer.
    TrainingArguments,                           # Class to define all training hyperparameters.
    Trainer,                                     # The core high-level training loop class.
    BitsAndBytesConfig                           # Configuration class for 8-bit/4-bit quantization.
)
from peft import (                               # Import Parameter-Efficient Fine-Tuning (PEFT) classes.
    LoraConfig,                                  # Configuration class for LoRA.
    get_peft_model,                              # Function to wrap the base model with LoRA adapters.
    prepare_model_for_kbit_training              # Function to prepare the model for 8-bit training.
)
from datasets import load_from_disk              # Function to load the processed dataset from disk.
from torch.utils.data import DataLoader          # Import DataLoader for managing data batches.

# Set the environment variable for deterministic CUDNN operations
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def load_config(config_path: str) -> dict:
    """
    Loads and parses the YAML configuration file.

    Args:
        config_path (str): The file path to the YAML configuration.

    Returns:
        dict: A dictionary containing the configuration parameters.
    """
    # Open the YAML file for reading
    with open(config_path, 'r') as f:
        # Load the content from the file
        config = yaml.safe_load(f)
    # Return the configuration dictionary
    return config

class SFTTrainer:
    """
    A supervised fine-tuning (SFT) trainer wrapper built on the Hugging Face Trainer.
    It integrates PEFT (LoRA) and bitsandbytes 8-bit quantization for resource-efficient training.
    """

    def __init__(self, config_path: str):
        """
        Initializes the trainer by loading configuration, setting up environment,
        and preparing the model and tokenizer.

        Args:
            config_path (str): Path to the training configuration YAML file.
        """
        # Load the configuration dictionary
        self.config = load_config(config_path)
        # Extract model name from config
        self.model_name = self.config['model_name']
        # Extract resource arguments
        self.resource_args = self.config['resource_args']
        # Extract training arguments
        self.training_args = self.config['training_args']
        # Explicitly disable TensorBoard reporting to prevent Windows pathing error
        # Set the 'report_to' argument to 'none' to only log locally (or not at all).
        self.training_args['report_to'] = 'none' 
        # Extract PEFT arguments
        self.peft_args = self.config['peft_args']
        # Set the global seed for reproducibility
        torch.manual_seed(self.training_args['seed'])
        
        # Robust Device Initialization
        device_string = self.resource_args['device'].lower()
        if torch.cuda.is_available() and device_string in ['cuda', 'gpu']:
            # Set to the first available CUDA device
            self.device = torch.device('cuda:0')
        else:
            # Fallback to CPU if no CUDA device is available or requested
            self.device = torch.device('cpu')

        # Prepare the model, tokenizer, and dataset
        self.model, self.tokenizer = self._setup_model_and_tokenizer()
        self.train_dataset = self._load_dataset()
        # Initialize the Hugging Face TrainingArguments instance
        self.args = TrainingArguments(**self.training_args)

    def _setup_model_and_tokenizer(self):
        """
        Loads the model with LoRA and 8-bit quantization if configured.

        Returns:
            tuple: A tuple containing the prepared model and its tokenizer.
        """
        # Load the tokenizer from the specified model name
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Set padding token to EOS token for GPT-style models
        tokenizer.pad_token = tokenizer.eos_token

        # Check if 8-bit optimization is requested
        if self.resource_args['use_8bit_optim']:
            # Define the BitsAndBytes configuration for 8-bit loading
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,                          # Quantize the model to 8-bit.
                llm_int8_threshold=6.0,                     # Outlier threshold for 8-bit quantization.
                llm_int8_skip_modules=['ln_f', 'lm_head']   # Modules to skip from 8-bit conversion.
            )
            # Load the model with 8-bit quantization configuration
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,             # Pass the 8-bit config.
                device_map={'': 0}
            )
            # Prepare the quantized model for k-bit training (recasts layernorm, embeds)
            model = prepare_model_for_kbit_training(model)
        else:
            # Load the model normally (e.g., in float32 for CPU or full precision for GPU)
            model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)

        # Configure LoRA using the PEFT parameters
        lora_config = LoraConfig(
            r=self.peft_args['lora_rank'],                # The rank of the update matrices (r).
            lora_alpha=self.peft_args['lora_alpha'],      # Scaling factor for LoRA updates.
            lora_dropout=self.peft_args['lora_dropout'],  # Dropout rate for LoRA.
            # Define the task type as Causal Language Modeling
            task_type="CAUSAL_LM", 
            # Modules (like linear layers) to apply LoRA to
            target_modules=self.peft_args['target_modules'], 
            bias="none"                                   # Only train the weight matrices, not bias.
        )

        # Wrap the base model with the LoRA configuration
        model = get_peft_model(model, lora_config)
        # Print the number of trainable parameters (only LoRA adapters are trainable)
        model.print_trainable_parameters()
        
        # Return the prepared model and tokenizer
        return model, tokenizer

    def _load_dataset(self):
        """
        Loads the preprocessed dataset and subsets it for the training job.

        Returns:
            datasets.Dataset: The tokenized dataset ready for training.
        """
        # Define the path where the tokenized dataset from M1 is saved
        dataset_path = 'data/tokenized/alpaca_processed'
        # Load the entire tokenized dataset from disk
        full_dataset = load_from_disk(dataset_path)
        
        # Get the requested size for this training run
        subset_size = self.config['dataset_size']
        # Take a small slice of the data for fast testing (Acceptance Criteria)
        train_subset = full_dataset['train'].select(range(subset_size))
        
        # Return the subsetted dataset
        return train_subset

    def train(self, resume_from_checkpoint: bool = False):
        """
        Initiates the supervised fine-tuning process.

        Args:
            resume_from_checkpoint (bool): Whether to resume from the last saved checkpoint.
        """

        # Explicitly create the output directory to resolve TensorBoard pathing errors on Windows
        output_dir = self.training_args['output_dir']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir}")

        # Determine the checkpoint path for resuming
        checkpoint_path = None
        if resume_from_checkpoint:
            # Check the output directory for existing checkpoints
            output_dir = self.training_args['output_dir']
            # Get a list of all checkpoint folders
            checkpoints = [
                os.path.join(output_dir, d) 
                for d in os.listdir(output_dir) 
                if d.startswith("checkpoint-")
            ]
            if checkpoints:
                # Find the most recent checkpoint folder
                checkpoint_path = max(checkpoints, key=os.path.getmtime)
                # Print a message indicating resumption
                print(f"Resuming training from checkpoint: {checkpoint_path}")

        # Initialize the Hugging Face Trainer
        trainer = Trainer(
            model=self.model,                       # The LoRA-wrapped, potentially 8-bit model.
            args=self.args,                         # The TrainingArguments instance.
            train_dataset=self.train_dataset,       # The training dataset subset.
            # Data collator is not strictly necessary for SFT if the data is already tokenized 
            # and padded/packed correctly, but we use the default for robustness.
        )
        
        # Set CUDNN to deterministic mode if requested for better reproducibility
        if self.resource_args['deterministic_dataloader'] and self.device.type == 'cuda':
            # Enable deterministic algorithms for CUDA (required for reproducibility)
            torch.use_deterministic_algorithms(True, warn_only=True)
            
        # Start the training process
        # The Trainer handles gradient accumulation, mixed precision, 
        # checkpoint saving (LoRA weights + optimizer state), and logging automatically.
        trainer.train(resume_from_checkpoint=checkpoint_path)

        # Save the final LoRA model weights
        final_save_path = os.path.join(self.training_args['output_dir'], "final_model")
        # Save the PEFT adapters and the tokenizer
        trainer.model.save_pretrained(final_save_path)
        self.tokenizer.save_pretrained(final_save_path)
        print(f"Final LoRA model saved to: {final_save_path}")