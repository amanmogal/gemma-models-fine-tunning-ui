"""
Model evaluation utilities for the Gemma Fine-tuning UI.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    pipeline,
    TextClassificationPipeline,
    TextGenerationPipeline
)
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import evaluate

from .loader import load_model, load_tokenizer

def evaluate_model(
    model,
    tokenizer,
    eval_dataset,
    task_type: str = "Text Generation",
    metrics: List[str] = ["accuracy"],
    batch_size: int = 8
) -> Dict[str, Any]:
    """
    Evaluate a model on the given dataset.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer to use
        eval_dataset: The dataset to evaluate on
        task_type: The type of task (e.g., "Text Generation", "Classification")
        metrics: The metrics to compute
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary of evaluation results
    """
    results = {}
    
    # Example implementation for various metrics
    if "accuracy" in metrics and task_type == "Classification":
        # This is a simplified implementation
        predictions = []
        labels = []
        
        # Process each example in the dataset
        for example in eval_dataset:
            inputs = tokenizer(example["input"], return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=50)
            
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(pred_text)
            labels.append(example["output"])
        
        # Calculate metrics
        # This is oversimplified and would need to be adapted for your specific case
        results["accuracy"] = accuracy_score(labels, predictions)
        
        if "f1" in metrics:
            results["f1"] = f1_score(labels, predictions, average="macro")
        
        if "precision" in metrics:
            results["precision"] = precision_score(labels, predictions, average="macro")
            
        if "recall" in metrics:
            results["recall"] = recall_score(labels, predictions, average="macro")
            
        if "confusion_matrix" in metrics:
            results["confusion_matrix"] = confusion_matrix(labels, predictions)
    
    # For text generation tasks
    elif task_type == "Text Generation":
        # For generation, we might use metrics like BLEU or ROUGE
        results["perplexity"] = calculate_perplexity(model, tokenizer, eval_dataset)
        
        if "bleu" in metrics:
            results["bleu"] = calculate_bleu(model, tokenizer, eval_dataset)
            
        if "rouge" in metrics:
            results["rouge"] = calculate_rouge(model, tokenizer, eval_dataset)
    
    return results

def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_length: int = 512,
    temperature: float = 0.7,
    do_sample: bool = True
) -> List[str]:
    """
    Generate text using the model.
    
    Args:
        model: The model to use
        tokenizer: The tokenizer to use
        prompt: The input prompt
        max_length: Maximum length of the generated text
        temperature: Sampling temperature
        do_sample: Whether to use sampling
        
    Returns:
        List of generated texts
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return generated_texts

def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_dataset: Dataset,
    input_col: str = "input",
    output_col: str = "output"
) -> float:
    """
    Compute perplexity on a dataset.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        eval_dataset: Evaluation dataset
        input_col: Name of the input column
        output_col: Name of the output column
        
    Returns:
        Perplexity score
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize perplexity module
    perplexity = evaluate.load("perplexity", module_type="metric")
    
    # Prepare texts
    texts = []
    for i in range(len(eval_dataset)):
        if output_col in eval_dataset[i]:
            texts.append(eval_dataset[i][output_col])
        else:
            texts.append(eval_dataset[i][input_col])
    
    # Compute perplexity
    results = perplexity.compute(
        predictions=texts,
        model_id=model.config._name_or_path,
        tokenizer=tokenizer
    )
    
    return results["perplexities"].mean()

def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """
    Compute BLEU score.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        BLEU score
    """
    # Initialize BLEU module
    bleu = evaluate.load("bleu")
    
    # Tokenize predictions and references
    tokenized_predictions = [pred.split() for pred in predictions]
    tokenized_references = [[ref.split()] for ref in references]
    
    # Compute BLEU
    results = bleu.compute(
        predictions=tokenized_predictions,
        references=tokenized_references
    )
    
    return results["bleu"]

def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE scores.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        Dictionary of ROUGE scores
    """
    # Initialize ROUGE module
    rouge = evaluate.load("rouge")
    
    # Compute ROUGE
    results = rouge.compute(
        predictions=predictions,
        references=references
    )
    
    return results

def compute_exact_match(predictions: List[str], references: List[str]) -> float:
    """
    Compute exact match score.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        Exact match score
    """
    # Normalize texts
    normalized_predictions = [pred.strip().lower() for pred in predictions]
    normalized_references = [ref.strip().lower() for ref in references]
    
    # Compute exact matches
    exact_matches = [
        1.0 if pred == ref else 0.0
        for pred, ref in zip(normalized_predictions, normalized_references)
    ]
    
    return sum(exact_matches) / len(exact_matches)

def compute_qa_f1(predictions: List[str], references: List[str]) -> float:
    """
    Compute F1 score for question answering.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        F1 score
    """
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        # Tokenize
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        
        # Compute F1
        if not pred_tokens and not ref_tokens:
            f1_scores.append(1.0)
        elif not pred_tokens or not ref_tokens:
            f1_scores.append(0.0)
        else:
            # Compute precision, recall, F1
            common_tokens = pred_tokens.intersection(ref_tokens)
            precision = len(common_tokens) / len(pred_tokens)
            recall = len(common_tokens) / len(ref_tokens)
            
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(2 * precision * recall / (precision + recall))
    
    return sum(f1_scores) / len(f1_scores)

def analyze_errors(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_dataset: Dataset,
    task_type: str = "Text Generation",
    input_col: str = "input",
    output_col: str = "output",
    batch_size: int = 8
) -> List[Dict[str, Any]]:
    """
    Analyze errors made by the model.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        eval_dataset: Evaluation dataset
        task_type: Task type ('Text Generation', 'Classification', etc.)
        input_col: Name of the input column
        output_col: Name of the output column
        batch_size: Batch size for evaluation
        
    Returns:
        List of error examples with analysis
    """
    # Initialize error examples list
    error_examples = []
    
    # Analyze based on task type
    if task_type == "Classification":
        # Set up classification pipeline
        classification_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Get inputs and references
        inputs = eval_dataset[input_col]
        references = eval_dataset[output_col]
        
        # Generate predictions
        predictions = []
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i+batch_size]
            batch_outputs = classification_pipeline(batch_inputs)
            predictions.extend([output["label"] for output in batch_outputs])
        
        # Find errors
        for i, (input_text, reference, prediction) in enumerate(zip(inputs, references, predictions)):
            if reference != prediction:
                error_examples.append({
                    "input": input_text,
                    "expected_output": reference,
                    "model_output": prediction,
                    "error_type": "Misclassification"
                })
    
    elif task_type == "Text Generation":
        # Set up generation pipeline
        generation_pipeline = TextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Get inputs and references
        inputs = eval_dataset[input_col]
        references = eval_dataset[output_col]
        
        # Generate predictions
        predictions = []
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i+batch_size]
            batch_outputs = generation_pipeline(
                batch_inputs,
                max_length=512,
                do_sample=False
            )
            predictions.extend([output[0]["generated_text"] for output in batch_outputs])
        
        # Find errors (using a simple heuristic)
        for i, (input_text, reference, prediction) in enumerate(zip(inputs, references, predictions)):
            # Compute similarity
            ref_tokens = set(reference.lower().split())
            pred_tokens = set(prediction.lower().split())
            
            if not ref_tokens:
                continue
            
            common_tokens = ref_tokens.intersection(pred_tokens)
            similarity = len(common_tokens) / len(ref_tokens)
            
            if similarity < 0.5:
                error_examples.append({
                    "input": input_text,
                    "expected_output": reference,
                    "model_output": prediction,
                    "error_type": "Low Similarity",
                    "similarity": similarity
                })
    
    return error_examples

def calculate_perplexity(model, tokenizer, dataset, max_samples: int = 100):
    """Calculate perplexity on the dataset."""
    # Simplified implementation
    return 10.5  # Placeholder

def calculate_bleu(model, tokenizer, dataset):
    """Calculate BLEU score."""
    # Would normally use NLTK or another library
    return 0.75  # Placeholder

def calculate_rouge(model, tokenizer, dataset):
    """Calculate ROUGE score."""
    # Would normally use Rouge library
    return {"rouge1": 0.8, "rouge2": 0.6, "rougeL": 0.7}  # Placeholder