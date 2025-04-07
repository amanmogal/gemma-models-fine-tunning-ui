import gradio as gr
import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple

def format_time(seconds: float) -> str:
    """Format seconds into a human-readable time string."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def update_status_html(
    status: str,
    current_step: int,
    total_steps: int,
    elapsed_time: float,
    remaining_time: Optional[float] = None
) -> str:
    """Update the status HTML with current training information."""
    if remaining_time is None:
        remaining_time_str = "N/A"
    else:
        remaining_time_str = format_time(remaining_time)
    
    return f"""
    <div style='padding: 10px; border-radius: 5px; background-color: #f0f0f0;'>
        <p><b>Status:</b> {status}</p>
        <p><b>Current step:</b> {current_step}/{total_steps}</p>
        <p><b>Time elapsed:</b> {format_time(elapsed_time)}</p>
        <p><b>Estimated time remaining:</b> {remaining_time_str}</p>
    </div>
    """

def create_error_message(message: str, severity: str = "error") -> str:
    """Create a formatted error message."""
    color = "#f44336" if severity == "error" else "#ff9800" if severity == "warning" else "#2196f3"
    
    return f"""
    <div style='padding: 10px; border-radius: 5px; background-color: {color}20; border: 1px solid {color}; margin-bottom: 10px;'>
        <p style='color: {color}; font-weight: bold;'>{severity.upper()}</p>
        <p>{message}</p>
    </div>
    """

def create_success_message(message: str) -> str:
    """Create a formatted success message."""
    return f"""
    <div style='padding: 10px; border-radius: 5px; background-color: #4caf5020; border: 1px solid #4caf50; margin-bottom: 10px;'>
        <p style='color: #4caf50; font-weight: bold;'>SUCCESS</p>
        <p>{message}</p>
    </div>
    """

def validate_hyperparameters(hyperparams: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate hyperparameters and return validation status and error messages."""
    errors = []
    
    # Check learning rate range
    lr = hyperparams.get("learning_rate", 0)
    if lr <= 0:
        errors.append("Learning rate must be greater than 0")
    elif lr > 1e-2:
        errors.append("Learning rate is too high (should be <= 0.01)")
    
    # Check batch size
    batch_size = hyperparams.get("batch_size", 0)
    if batch_size <= 0:
        errors.append("Batch size must be greater than 0")
    elif batch_size > 128:
        errors.append("Batch size is too large (should be <= 128)")
    
    # Check epochs
    epochs = hyperparams.get("epochs", 0)
    if epochs <= 0:
        errors.append("Number of epochs must be greater than 0")
    elif epochs > 100:
        errors.append("Number of epochs is too high (should be <= 100)")
    
    # Check gradient accumulation steps
    grad_accum = hyperparams.get("gradient_accumulation_steps", 0)
    if grad_accum <= 0:
        errors.append("Gradient accumulation steps must be greater than 0")
    elif grad_accum > 32:
        errors.append("Gradient accumulation steps is too high (should be <= 32)")
    
    # Check weight decay
    weight_decay = hyperparams.get("weight_decay", 0)
    if weight_decay < 0:
        errors.append("Weight decay must be non-negative")
    elif weight_decay > 0.1:
        errors.append("Weight decay is too high (should be <= 0.1)")
    
    # Check warmup ratio
    warmup = hyperparams.get("warmup_ratio", 0)
    if warmup < 0:
        errors.append("Warmup ratio must be non-negative")
    elif warmup > 0.5:
        errors.append("Warmup ratio is too high (should be <= 0.5)")
    
    # Check optimizer
    optimizer = hyperparams.get("optimizer", "").lower()
    if optimizer not in ["adamw", "adam8bit", "lion"]:
        errors.append("Invalid optimizer choice")
    
    # Check learning rate scheduler
    lr_scheduler = hyperparams.get("lr_scheduler", "").lower()
    if lr_scheduler not in ["linear", "cosine", "cosine_with_restarts", "polynomial"]:
        errors.append("Invalid learning rate scheduler choice")
    
    return len(errors) == 0, errors

def format_metrics_for_display(metrics: Dict[str, float]) -> List[List[Any]]:
    """Format metrics dictionary into a list for DataFrame display."""
    return [[k, round(v, 4)] for k, v in metrics.items()]

def get_theme() -> gr.Theme:
    """Get a custom theme for the Gradio interface."""
    return gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        neutral_hue="slate",
        radius_size=gr.themes.sizes.radius_sm,
        text_size=gr.themes.sizes.text_md
    )

def save_config_to_json(config: Dict[str, Any], filename: str = "config.json") -> str:
    """Save configuration to a JSON file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, "w") as f:
        json.dump(config, f, indent=2)
    
    return filename

def load_config_from_json(filename: str = "config.json") -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    if not os.path.exists(filename):
        return {}
    
    with open(filename, "r") as f:
        return json.load(f)

def format_file_size(size_bytes: int) -> str:
    """Format file size in bytes to a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def estimate_memory_requirements(
    model_name: str,
    fine_tuning_method: str,
    batch_size: int,
    sequence_length: int
) -> Dict[str, Any]:
    """Estimate memory requirements for training."""
    # Base memory requirements in GB
    base_memory = {
        "gemma-2b": 4,
        "gemma-7b": 14
    }
    
    # Method multipliers
    method_multiplier = {
        "Full Fine-tuning": 3.0,
        "LoRA": 1.2,
        "QLoRA": 0.8,
        "Adapter": 1.1
    }
    
    # Calculate memory requirements
    model_memory = base_memory.get(model_name, 10)
    method_factor = method_multiplier.get(fine_tuning_method, 1.0)
    sequence_factor = sequence_length / 512  # Normalize to 512 tokens
    batch_factor = batch_size / 8  # Normalize to batch size 8
    
    training_memory = model_memory * method_factor * sequence_factor * batch_factor
    
    return {
        "model_base": model_memory,
        "training_estimated": training_memory,
        "recommended_gpu": "16GB+ GPU" if training_memory > 12 else "8GB+ GPU" if training_memory > 6 else "4GB+ GPU"
    } 