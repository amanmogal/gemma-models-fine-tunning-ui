"""
Model training utilities for the Gemma Fine-tuning UI.
"""

import os
import torch
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from datasets import Dataset, DatasetDict
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

from .loader import load_model, load_tokenizer

logger = logging.getLogger(__name__)

class TrainingProgressCallback(TrainerCallback):
    """Callback to track and report training progress."""
    
    def __init__(
        self,
        progress_update_fn: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Initialize the callback.
        
        Args:
            progress_update_fn: Function to call with progress updates
        """
        self.progress_update_fn = progress_update_fn
        self.start_time = None
        self.step_times = []
    
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Called when training begins."""
        self.start_time = time.time()
        
        if self.progress_update_fn:
            self.progress_update_fn({
                "status": "started",
                "current_step": 0,
                "total_steps": state.max_steps,
                "elapsed_time": 0,
                "remaining_time": None,
                "loss": None
            })
    
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Called at the end of each step."""
        if not self.start_time:
            self.start_time = time.time()
        
        # Calculate elapsed time
        elapsed_time = time.time() - self.start_time
        
        # Track step time for estimating remaining time
        if len(self.step_times) >= 20:
            self.step_times.pop(0)
        self.step_times.append(time.time())
        
        # Estimate remaining time
        remaining_time = None
        if len(self.step_times) >= 2 and state.global_step > 0:
            avg_step_time = (self.step_times[-1] - self.step_times[0]) / (len(self.step_times) - 1)
            remaining_steps = state.max_steps - state.global_step
            remaining_time = avg_step_time * remaining_steps
        
        # Get current loss
        loss = None
        if state.log_history:
            for entry in reversed(state.log_history):
                if "loss" in entry:
                    loss = entry["loss"]
                    break
        
        if self.progress_update_fn:
            self.progress_update_fn({
                "status": "training",
                "current_step": state.global_step,
                "total_steps": state.max_steps,
                "elapsed_time": elapsed_time,
                "remaining_time": remaining_time,
                "loss": loss
            })
    
    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Called when training ends."""
        elapsed_time = time.time() - self.start_time
        
        if self.progress_update_fn:
            self.progress_update_fn({
                "status": "completed",
                "current_step": state.global_step,
                "total_steps": state.max_steps,
                "elapsed_time": elapsed_time,
                "remaining_time": 0,
                "loss": state.log_history[-1].get("loss") if state.log_history else None
            })

def create_training_args(
    output_dir: str = "./results",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    lr_scheduler_type: str = "linear",
    evaluation_strategy: str = "steps",
    eval_steps: int = 500,
    save_strategy: str = "steps",
    save_steps: int = 500,
    logging_steps: int = 100,
    fp16: bool = True,
    bf16: bool = False,
    optim: str = "adamw_torch",
    early_stopping_patience: Optional[int] = None,
    max_grad_norm: float = 1.0,
    seed: int = 42
) -> TrainingArguments:
    """
    Create training arguments for the Trainer.
    
    Args:
        output_dir: Directory to save model checkpoints
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device during training
        per_device_eval_batch_size: Batch size per device during evaluation
        gradient_accumulation_steps: Number of steps to accumulate gradients
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_ratio: Ratio of steps for learning rate warmup
        lr_scheduler_type: Learning rate scheduler type
        evaluation_strategy: When to evaluate
        eval_steps: Number of steps between evaluations
        save_strategy: When to save checkpoints
        save_steps: Number of steps between checkpoint saves
        logging_steps: Number of steps between logging
        fp16: Whether to use 16-bit floating point precision
        bf16: Whether to use bfloat16 precision
        optim: Optimizer to use
        early_stopping_patience: Patience for early stopping
        max_grad_norm: Maximum gradient norm
        seed: Random seed
        
    Returns:
        TrainingArguments object
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        logging_steps=logging_steps,
        fp16=fp16 and not bf16,  # Don't use both
        bf16=bf16,
        optim=optim,
        max_grad_norm=max_grad_norm,
        seed=seed,
        load_best_model_at_end=True if early_stopping_patience else False,
        report_to="tensorboard"
    )
    
    return training_args

def setup_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None
) -> LoraConfig:
    """
    Set up a LoRA configuration.
    
    Args:
        r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        target_modules: List of modules to apply LoRA to
        
    Returns:
        LoRA configuration
    """
    # Default target modules for Gemma if not specified
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

def train_model(
    model_id: str,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    fine_tuning_method: str = "LoRA",
    training_args: Optional[TrainingArguments] = None,
    lora_config: Optional[LoraConfig] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    device_map: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a Gemma model.
    
    Args:
        model_id: Model identifier
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        fine_tuning_method: Fine-tuning method ('Full Fine-tuning', 'LoRA', 'QLoRA', 'Adapter')
        training_args: Training arguments
        lora_config: LoRA configuration
        progress_callback: Function to call with progress updates
        device_map: Device mapping strategy
        load_in_8bit: Whether to load the model in 8-bit precision
        load_in_4bit: Whether to load the model in 4-bit precision
        
    Returns:
        Tuple of (trained model, training metrics)
    """
    # Load tokenizer
    tokenizer = load_tokenizer(model_id)
    
    # Load model
    model = load_model(
        model_id,
        device_map=device_map,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit
    )
    
    # Apply fine-tuning method
    if fine_tuning_method == "LoRA" or fine_tuning_method == "QLoRA":
        # Prepare model for k-bit training if using quantization
        if load_in_8bit or load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        
        # Set up LoRA config if not provided
        if lora_config is None:
            lora_config = setup_lora_config()
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Set up default training arguments if not provided
    if training_args is None:
        training_args = create_training_args()
    
    # Set up data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Set up callbacks
    callbacks = []
    
    # Add early stopping if specified
    if training_args.load_best_model_at_end:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=3
        ))
    
    # Add progress tracking callback
    if progress_callback:
        callbacks.append(TrainingProgressCallback(progress_callback))
    
    # Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks
    )
    
    # Train model
    train_result = trainer.train()
    
    # Save model
    trainer.save_model()
    
    # Save tokenizer
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Get training metrics
    metrics = train_result.metrics
    
    # Add evaluation metrics if available
    if eval_dataset:
        eval_metrics = trainer.evaluate()
        metrics.update(eval_metrics)
    
    # Log and save metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    return model, metrics 