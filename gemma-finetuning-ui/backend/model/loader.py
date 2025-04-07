"""
Model loading utilities for the Gemma Fine-tuning UI.
"""

import os
import torch
from typing import Dict, List, Any, Optional, Tuple, Union
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import PeftModel, PeftConfig

def get_available_models() -> List[Dict[str, Any]]:
    """
    Get a list of available Gemma models.
    
    Returns:
        List of model information dictionaries
    """
    return [
        {
            "id": "google/gemma-2b",
            "name": "Gemma 2B",
            "description": "2 billion parameter base model",
            "parameters": "2B",
            "context_length": 8192,
            "requires_api_key": True
        },
        {
            "id": "google/gemma-2b-it",
            "name": "Gemma 2B-it",
            "description": "2 billion parameter instruction-tuned model",
            "parameters": "2B",
            "context_length": 8192,
            "requires_api_key": True
        },
        {
            "id": "google/gemma-7b",
            "name": "Gemma 7B",
            "description": "7 billion parameter base model",
            "parameters": "7B",
            "context_length": 8192,
            "requires_api_key": True
        },
        {
            "id": "google/gemma-7b-it",
            "name": "Gemma 7B-it",
            "description": "7 billion parameter instruction-tuned model",
            "parameters": "7B",
            "context_length": 8192,
            "requires_api_key": True
        }
    ]

def load_tokenizer(
    model_id: str,
    use_fast: bool = True
) -> PreTrainedTokenizer:
    """
    Load a tokenizer for a Gemma model.
    
    Args:
        model_id: Model identifier
        use_fast: Whether to use the fast tokenizer implementation
        
    Returns:
        Loaded tokenizer
    """
    # Check if API key is set
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Google API key not found. Please set the GOOGLE_API_KEY environment variable."
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=use_fast,
        token=api_key
    )
    
    # Ensure the tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

def load_model(
    model_id: str,
    device_map: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    adapter_path: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None
) -> PreTrainedModel:
    """
    Load a Gemma model.
    
    Args:
        model_id: Model identifier
        device_map: Device mapping strategy
        load_in_8bit: Whether to load the model in 8-bit precision
        load_in_4bit: Whether to load the model in 4-bit precision
        adapter_path: Path to a PEFT adapter to load
        torch_dtype: PyTorch data type for model weights
        
    Returns:
        Loaded model
    """
    # Check if API key is set
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Google API key not found. Please set the GOOGLE_API_KEY environment variable."
        )
    
    # Configure quantization if requested
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    
    # Set default torch dtype if not specified
    if torch_dtype is None:
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        token=api_key
    )
    
    # Load adapter if specified
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    
    return model

def check_model_compatibility(
    model_id: str,
    task_type: str
) -> Tuple[bool, str]:
    """
    Check if a model is compatible with a specific task.
    
    Args:
        model_id: Model identifier
        task_type: Task type ('Text Generation', 'Classification', etc.)
        
    Returns:
        Tuple of (is_compatible, message)
    """
    # All Gemma models are compatible with text generation
    if task_type == 'Text Generation':
        return True, "Model is compatible with text generation tasks."
    
    # For classification, instruction-tuned models are preferred
    if task_type == 'Classification':
        if 'it' in model_id:
            return True, "Instruction-tuned model is well-suited for classification tasks."
        else:
            return True, "Base model can be used for classification, but instruction-tuned models may perform better."
    
    # For question answering, instruction-tuned models are preferred
    if task_type == 'Question Answering':
        if 'it' in model_id:
            return True, "Instruction-tuned model is well-suited for question answering tasks."
        else:
            return True, "Base model can be used for question answering, but instruction-tuned models may perform better."
    
    # For summarization, larger models are preferred
    if task_type == 'Summarization':
        if '7b' in model_id.lower():
            return True, "7B model is well-suited for summarization tasks."
        else:
            return True, "2B model can be used for summarization, but larger models may perform better."
    
    # Default: assume compatibility
    return True, "Model is assumed to be compatible with the selected task."

def get_model_info(model_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific model.
    
    Args:
        model_id: Model identifier
        
    Returns:
        Dictionary with model information
    """
    available_models = get_available_models()
    
    for model in available_models:
        if model["id"] == model_id:
            return model
    
    # If model not found, return basic info
    return {
        "id": model_id,
        "name": model_id.split("/")[-1],
        "description": "Custom model",
        "parameters": "Unknown",
        "context_length": 8192,
        "requires_api_key": True
    } 