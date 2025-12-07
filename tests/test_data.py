import pytest
from data.preprocessor import TokenizerWrapper, Preprocessor
from datasets import Dataset
from typing import Dict, Any

# --- MOCK SETUP ---
def mock_load_config() -> Dict[str, Any]:
    """Provides a minimal configuration for initializing data classes."""
    return {
        'model_config': {'model_name': 'distilgpt2'},
        'preproc_config': {
            'max_seq_length': 128, 
            'seed': 42,
            'alpaca_template': 'Instruction: {instruction}\nInput: {input}\nResponse: {output}'}
    }

@pytest.fixture(scope="module")
def tokenizer_wrapper():
    """Fixture to provide a TokenizerWrapper instance for tests."""
    config = mock_load_config()
    
    # Explicitly pass the required arguments to the constructor
    return TokenizerWrapper(
        model_name=config['model_config']['model_name'],
        max_length=config['preproc_config']['max_seq_length'], 
        seed=config['preproc_config']['seed']
    )

@pytest.fixture(scope="module")
def preprocessor():
    """Fixture to provide a Preprocessor instance for tests."""
    return Preprocessor(mock_load_config())

# --- UNIT TESTS ---

def test_tokenizer_roundtrip(tokenizer_wrapper: TokenizerWrapper):
    """
    Verifies that tokenizing a string and decoding the result yields the 
    original string, checking for tokenization symmetry.

    This test is a fundamental check for tokenization symmetry. 
    When we convert human-readable text to numeric tokens and back again, 
    we must ensure we don't lose any information or introduce artifacts. 
    If the decoded string doesn't match the original, the model will be 
    trained on corrupted or truncated data, which leads to poor performance.
    """
    test_string = "The quick brown fox jumps over the lazy dog. This is a critical check for LLM data integrity."
    
    # 1. Tokenize the string
    tokenized_output = tokenizer_wrapper.tokenizer(
        test_string, 
        truncation=True, 
        max_length=tokenizer_wrapper.max_length
    )
    input_ids = tokenized_output['input_ids']
    
    # 2. Decode the token IDs back into a string
    decoded_string = tokenizer_wrapper.tokenizer.decode(input_ids, skip_special_tokens=True)
    
    # 3. Assert the strings are equivalent (ignoring leading/trailing whitespace)
    assert decoded_string.strip() == test_string.strip()

def test_preprocess_no_nulls(preprocessor: Preprocessor):
    """
    Verifies that the preprocessor's integrity check filters out malformed inputs 
    (e.g., nulls or empty strings for instruction/output).

    This test validates the Data Integrity and Fault Tolerance. Machine learning 
    pipelines must be robust to malformed data from raw sources. This test specifically 
    ensures that the Preprocessor's filtering logic (check_integrity) correctly identifies 
    and drops examples that contain None or empty strings in critical fields 
    (like instruction or output) after the formatting step. This prevents the training 
    process from crashing or from learning from meaningless examples.
    """
    # Define a mock dataset with examples that should pass and fail integrity checks
    raw_data = [
        # Good example: All fields present
        {'instruction': 'Write a poem', 'input': '', 'output': 'Roses are red...'}, 
        # Bad example 1: Empty output (should be filtered)
        {'instruction': 'What is your name?', 'input': '', 'output': ''},           
        # Bad example 2: Null instruction (should be filtered)
        {'instruction': None, 'input': 'N/A', 'output': 'This should fail'},        
        # Good example: All fields present
        {'instruction': 'Answer this', 'input': 'N/A', 'output': 'Valid output'}    
    ]
    
    # Create a Hugging Face Dataset object
    mock_dataset = Dataset.from_list(raw_data)

    # 1. Apply SFT formatting (which creates the 'text' column from instruction/output)
    formatted_dataset = mock_dataset.map(preprocessor.format_alpaca_example)

    # 2. Apply the filtering based on the 'text' column content
    filtered_dataset = formatted_dataset.filter(preprocessor.check_integrity)

    # 3. Verify the count: We expect 2 good examples to remain
    expected_count = 2
    
    assert len(filtered_dataset) == expected_count
    
    # Optional: Verify the content to be sure the correct examples were kept
    filtered_texts = filtered_dataset['text']
    assert 'Roses are red...' in filtered_texts[0]
    assert 'Valid output' in filtered_texts[1]