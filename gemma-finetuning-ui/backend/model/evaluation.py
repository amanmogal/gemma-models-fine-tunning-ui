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
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    eval_dataset: Dataset,
    task_type: str = "Text Generation",
    input_col: str = "input",
    output_col: str = "output",
    metrics: Optional[List[str]] = None,
    batch_size: int = 8,
    max_length: int = 512
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        eval_dataset: Evaluation dataset
        task_type: Task type ('Text Generation', 'Classification', etc.)
        input_col: Name of the input column
        output_col: Name of the output column
        metrics: List of metrics to compute
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Set default metrics based on task type
    if metrics is None:
        if task_type == "Text Generation":
            metrics = ["perplexity", "bleu", "rouge"]
        elif task_type == "Classification":
            metrics = ["accuracy", "f1", "precision", "recall"]
        elif task_type == "Question Answering":
            metrics = ["exact_match", "f1"]
        else:
            metrics = ["perplexity"]
    
    # Initialize results dictionary
    results = {}
    
    # Evaluate based on task type
    if task_type == "Text Generation":
        # Set up generation pipeline
        generation_pipeline = TextGenerationPipeline(
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Generate outputs
        inputs = eval_dataset[input_col]
        references = eval_dataset[output_col]
        
        # Generate predictions
        predictions = []
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i+batch_size]
            batch_outputs = generation_pipeline(
                batch_inputs,
                max_length=max_length,
                do_sample=False
            )
            predictions.extend([output[0]["generated_text"] for output in batch_outputs])
        
        # Compute metrics
        if "perplexity" in metrics:
            perplexity = compute_perplexity(model, tokenizer, eval_dataset, input_col, output_col)
            results["perplexity"] = perplexity
        
        if "bleu" in metrics:
            bleu = compute_bleu(predictions, references)
            results["bleu"] = bleu
        
        if "rouge" in metrics:
            rouge_scores = compute_rouge(predictions, references)
            results.update(rouge_scores)
    
    elif task_type == "Classification":
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
        
        # Compute metrics
        if "accuracy" in metrics:
            accuracy = accuracy_score(references, predictions)
            results["accuracy"] = accuracy
        
        if any(m in metrics for m in ["f1", "precision", "recall"]):
            precision, recall, f1, _ = precision_recall_fscore_support(
                references, predictions, average="weighted"
            )
            
            if "precision" in metrics:
                results["precision"] = precision
            
            if "recall" in metrics:
                results["recall"] = recall
            
            if "f1" in metrics:
                results["f1"] = f1
        
        if "confusion_matrix" in metrics:
            cm = confusion_matrix(references, predictions)
            results["confusion_matrix"] = cm.tolist()
    
    elif task_type == "Question Answering":
        # Set up QA pipeline
        qa_pipeline = pipeline(
            "question-answering",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Get inputs and references
        questions = eval_dataset[input_col]
        references = eval_dataset[output_col]
        
        # Generate predictions
        predictions = []
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i+batch_size]
            batch_outputs = qa_pipeline(batch_questions)
            predictions.extend([output["answer"] for output in batch_outputs])
        
        # Compute metrics
        if "exact_match" in metrics:
            exact_match = compute_exact_match(predictions, references)
            results["exact_match"] = exact_match
        
        if "f1" in metrics:
            f1 = compute_qa_f1(predictions, references)
            results["f1"] = f1
    
    return results

def generate_text(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    num_return_sequences: int = 1,
    do_sample: bool = True
) -> List[str]:
    """
    Generate text from a prompt.
    
    Args:
        model: Model to use for generation
        tokenizer: Tokenizer for the model
        prompt: Input prompt
        max_length: Maximum sequence length
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        num_return_sequences: Number of sequences to generate
        do_sample: Whether to use sampling
        
    Returns:
        List of generated texts
    """
    # Set up generation pipeline
    generation_pipeline = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Generate text
    outputs = generation_pipeline(
        prompt,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample
    )
    
    # Extract generated texts
    generated_texts = [output["generated_text"] for output in outputs]
    
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