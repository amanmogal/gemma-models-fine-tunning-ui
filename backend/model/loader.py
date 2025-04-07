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

def get_available_models() -> List[str]:
    """Return a list of available Gemma models."""
    return ["gemma-2b", "gemma-7b", "gemma-2b-it", "gemma-7b-it"]

def load_tokenizer(model_id: str):
    """Load tokenizer for the specified model."""
    if "gemma" in model_id.lower():
        model_path = f"google/{model_id}"
    else:
        model_path = model_id
        
    return AutoTokenizer.from_pretrained(model_path)

def load_model(
    model_id: str,
    device_map: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
):
    """
    Load a pre-trained model.
    
    Args:
        model_id: ID or path of the model to load
        device_map: Device mapping strategy
        load_in_8bit: Whether to load in 8-bit precision
        load_in_4bit: Whether to load in 4-bit precision
    
    Returns:
        The loaded model
    """
    if "gemma" in model_id.lower():
        model_path = f"google/{model_id}"
    else:
        model_path = model_id
    
    # Set quantization options
    quantization_config = None
    
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    elif load_in_8bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
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

def get_api_credentials():
    """Get API credentials from environment or configuration file."""
    api_key = os.environ.get("HF_TOKEN") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # Load from .env file if possible
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("HF_TOKEN") or os.environ.get("GOOGLE_API_KEY")
        except ImportError:
            pass
    return api_key