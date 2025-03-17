"""
Configuration utilities for the Gemma Fine-tuning UI.
"""

import os
import json
from typing import Dict, Any, Optional

# Default configuration
DEFAULT_CONFIG = {
    "app": {
        "title": "Gemma Fine-tuning UI",
        "description": "A user-friendly interface for fine-tuning Google's Gemma language models",
        "theme": "light",
        "debug": False
    },
    "models": {
        "default_model": "google/gemma-2b",
        "cache_dir": "./model_cache"
    },
    "training": {
        "output_dir": "./results",
        "default_batch_size": 8,
        "default_learning_rate": 2e-5,
        "default_epochs": 3,
        "default_method": "LoRA",
        "tensorboard_dir": "./logs"
    },
    "data": {
        "upload_dir": "./uploads",
        "cache_dir": "./data_cache",
        "examples_dir": "./examples"
    },
    "export": {
        "export_dir": "./exported_models",
        "formats": ["HuggingFace", "ONNX", "TensorFlow", "PyTorch", "GGUF"]
    }
}

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from a JSON file, or create a default one if it doesn't exist.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    # If config file exists, load it
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            
            # Merge with default config to ensure all keys exist
            merged_config = DEFAULT_CONFIG.copy()
            _deep_update(merged_config, config)
            return merged_config
        except Exception as e:
            print(f"Error loading config: {e}")
            return DEFAULT_CONFIG
    
    # Otherwise, create a default config
    save_config(DEFAULT_CONFIG, config_path)
    return DEFAULT_CONFIG

def save_config(config: Dict[str, Any], config_path: str = "config.json") -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to the configuration file
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path) if os.path.dirname(config_path) else ".", exist_ok=True)
        
        # Save config
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Error saving config: {e}")

def get_config_value(
    config: Dict[str, Any],
    key_path: str,
    default: Any = None
) -> Any:
    """
    Get a value from the configuration using a dot-separated key path.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the key (e.g., "training.output_dir")
        default: Default value to return if the key doesn't exist
        
    Returns:
        Value from the configuration, or the default if not found
    """
    keys = key_path.split(".")
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value

def set_config_value(
    config: Dict[str, Any],
    key_path: str,
    value: Any
) -> Dict[str, Any]:
    """
    Set a value in the configuration using a dot-separated key path.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the key (e.g., "training.output_dir")
        value: Value to set
        
    Returns:
        Updated configuration dictionary
    """
    keys = key_path.split(".")
    current = config
    
    # Navigate to the nested dictionary
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the value
    current[keys[-1]] = value
    
    return config

def _deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep update a nested dictionary with another dictionary.
    
    Args:
        target: Target dictionary to update
        source: Source dictionary with updates
        
    Returns:
        Updated target dictionary
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    
    return target

def create_directory_structure(config: Dict[str, Any]) -> None:
    """
    Create the directory structure based on the configuration.
    
    Args:
        config: Configuration dictionary
    """
    # Create directories
    directories = [
        get_config_value(config, "models.cache_dir"),
        get_config_value(config, "training.output_dir"),
        get_config_value(config, "training.tensorboard_dir"),
        get_config_value(config, "data.upload_dir"),
        get_config_value(config, "data.cache_dir"),
        get_config_value(config, "data.examples_dir"),
        get_config_value(config, "export.export_dir")
    ]
    
    for directory in directories:
        if directory:
            os.makedirs(directory, exist_ok=True) 