"""
Dataset validation utilities for the Gemma Fine-tuning UI.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datasets import Dataset, DatasetDict

def validate_sample(
    input_text: str,
    output_text: Optional[str] = None,
    min_input_length: int = 1,
    max_input_length: int = 2048,
    min_output_length: int = 0,
    max_output_length: int = 2048
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Validate a single sample for potential issues.
    
    Args:
        input_text: Input text
        output_text: Output text (optional)
        min_input_length: Minimum input length
        max_input_length: Maximum input length
        min_output_length: Minimum output length
        max_output_length: Maximum output length
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    
    # Check if input is empty or None
    if not input_text or not isinstance(input_text, str):
        issues.append({
            "issue": "Empty input",
            "severity": "Error",
            "description": "Input text is empty or not a string"
        })
    else:
        # Check input length
        if len(input_text) < min_input_length:
            issues.append({
                "issue": "Input too short",
                "severity": "Warning",
                "description": f"Input text is shorter than {min_input_length} characters"
            })
        
        if len(input_text) > max_input_length:
            issues.append({
                "issue": "Input too long",
                "severity": "Warning",
                "description": f"Input text is longer than {max_input_length} characters and may be truncated"
            })
    
    # Check output if provided
    if output_text is not None:
        if not isinstance(output_text, str):
            issues.append({
                "issue": "Invalid output",
                "severity": "Error",
                "description": "Output text is not a string"
            })
        elif len(output_text) < min_output_length:
            issues.append({
                "issue": "Output too short",
                "severity": "Warning",
                "description": f"Output text is shorter than {min_output_length} characters"
            })
        
        if len(output_text) > max_output_length:
            issues.append({
                "issue": "Output too long",
                "severity": "Warning",
                "description": f"Output text is longer than {max_output_length} characters and may be truncated"
            })
    
    return len(issues) == 0, issues

def validate_dataset(
    dataset: Dataset,
    input_col: str = 'input',
    output_col: str = 'output',
    min_input_length: int = 1,
    max_input_length: int = 2048,
    min_output_length: int = 0,
    max_output_length: int = 2048,
    check_duplicates: bool = True,
    sample_size: Optional[int] = 100
) -> Tuple[bool, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Validate a dataset for potential issues.
    
    Args:
        dataset: Dataset to validate
        input_col: Name of the input column
        output_col: Name of the output column
        min_input_length: Minimum input length
        max_input_length: Maximum input length
        min_output_length: Minimum output length
        max_output_length: Maximum output length
        check_duplicates: Whether to check for duplicates
        sample_size: Number of samples to check (None for all)
        
    Returns:
        Tuple of (is_valid, list of issues, validation stats)
    """
    issues = []
    stats = {
        "total_samples": len(dataset),
        "valid_samples": 0,
        "invalid_samples": 0,
        "issues_by_type": {}
    }
    
    # Convert to pandas for easier analysis
    if sample_size is not None and sample_size < len(dataset):
        # Sample a subset for validation
        indices = np.random.choice(len(dataset), sample_size, replace=False)
        df = dataset.select(indices).to_pandas()
    else:
        df = dataset.to_pandas()
    
    # Check for missing columns
    if input_col not in df.columns:
        issues.append({
            "issue": "Missing input column",
            "severity": "Error",
            "description": f"Input column '{input_col}' not found in dataset"
        })
        return False, issues, stats
    
    if output_col not in df.columns:
        issues.append({
            "issue": "Missing output column",
            "severity": "Warning",
            "description": f"Output column '{output_col}' not found in dataset"
        })
    
    # Check for missing values
    input_null_count = df[input_col].isnull().sum()
    if input_null_count > 0:
        issues.append({
            "issue": "Missing input values",
            "severity": "Error",
            "description": f"{input_null_count} samples have missing input values"
        })
        stats["issues_by_type"]["missing_input"] = input_null_count
    
    if output_col in df.columns:
        output_null_count = df[output_col].isnull().sum()
        if output_null_count > 0:
            issues.append({
                "issue": "Missing output values",
                "severity": "Warning",
                "description": f"{output_null_count} samples have missing output values"
            })
            stats["issues_by_type"]["missing_output"] = output_null_count
    
    # Check for duplicates
    if check_duplicates:
        duplicate_count = len(df) - len(df.drop_duplicates(subset=[input_col]))
        if duplicate_count > 0:
            issues.append({
                "issue": "Duplicate inputs",
                "severity": "Warning",
                "description": f"{duplicate_count} samples have duplicate input values"
            })
            stats["issues_by_type"]["duplicates"] = duplicate_count
    
    # Check input lengths
    if input_col in df.columns:
        df['input_length'] = df[input_col].astype(str).str.len()
        
        short_inputs = (df['input_length'] < min_input_length).sum()
        if short_inputs > 0:
            issues.append({
                "issue": "Short inputs",
                "severity": "Warning",
                "description": f"{short_inputs} samples have input text shorter than {min_input_length} characters"
            })
            stats["issues_by_type"]["short_inputs"] = short_inputs
        
        long_inputs = (df['input_length'] > max_input_length).sum()
        if long_inputs > 0:
            issues.append({
                "issue": "Long inputs",
                "severity": "Warning",
                "description": f"{long_inputs} samples have input text longer than {max_input_length} characters and may be truncated"
            })
            stats["issues_by_type"]["long_inputs"] = long_inputs
    
    # Check output lengths
    if output_col in df.columns:
        df['output_length'] = df[output_col].astype(str).str.len()
        
        short_outputs = (df['output_length'] < min_output_length).sum()
        if short_outputs > 0:
            issues.append({
                "issue": "Short outputs",
                "severity": "Warning",
                "description": f"{short_outputs} samples have output text shorter than {min_output_length} characters"
            })
            stats["issues_by_type"]["short_outputs"] = short_outputs
        
        long_outputs = (df['output_length'] > max_output_length).sum()
        if long_outputs > 0:
            issues.append({
                "issue": "Long outputs",
                "severity": "Warning",
                "description": f"{long_outputs} samples have output text longer than {max_output_length} characters and may be truncated"
            })
            stats["issues_by_type"]["long_outputs"] = long_outputs
    
    # Calculate valid/invalid samples
    invalid_count = 0
    for issue in issues:
        if issue["severity"] == "Error":
            if "samples" in issue["description"]:
                # Extract the number from the description
                try:
                    count = int(issue["description"].split(" ")[0])
                    invalid_count = max(invalid_count, count)
                except:
                    pass
    
    stats["invalid_samples"] = invalid_count
    stats["valid_samples"] = stats["total_samples"] - invalid_count
    
    # Add length statistics
    if 'input_length' in df.columns:
        stats["input_length_mean"] = df['input_length'].mean()
        stats["input_length_median"] = df['input_length'].median()
        stats["input_length_min"] = df['input_length'].min()
        stats["input_length_max"] = df['input_length'].max()
    
    if 'output_length' in df.columns:
        stats["output_length_mean"] = df['output_length'].mean()
        stats["output_length_median"] = df['output_length'].median()
        stats["output_length_min"] = df['output_length'].min()
        stats["output_length_max"] = df['output_length'].max()
    
    return len([i for i in issues if i["severity"] == "Error"]) == 0, issues, stats

def validate_dataset_for_task(
    dataset: Dataset,
    task_type: str,
    input_col: str = 'input',
    output_col: str = 'output'
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Validate a dataset for a specific task type.
    
    Args:
        dataset: Dataset to validate
        task_type: Task type ('Text Generation', 'Classification', etc.)
        input_col: Name of the input column
        output_col: Name of the output column
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    
    # Convert to pandas for easier analysis
    df = dataset.to_pandas()
    
    if task_type == 'Classification':
        # For classification, check if output values are consistent
        if output_col in df.columns:
            unique_labels = df[output_col].unique()
            
            if len(unique_labels) < 2:
                issues.append({
                    "issue": "Insufficient classes",
                    "severity": "Error",
                    "description": f"Classification task requires at least 2 classes, but only {len(unique_labels)} found"
                })
            
            # Check class balance
            class_counts = df[output_col].value_counts()
            min_class_count = class_counts.min()
            max_class_count = class_counts.max()
            
            if min_class_count < 5:
                issues.append({
                    "issue": "Small class",
                    "severity": "Warning",
                    "description": f"Smallest class has only {min_class_count} samples"
                })
            
            if min_class_count / max_class_count < 0.1:
                issues.append({
                    "issue": "Class imbalance",
                    "severity": "Warning",
                    "description": f"Severe class imbalance detected (ratio: {min_class_count / max_class_count:.2f})"
                })
    
    elif task_type == 'Text Generation':
        # For text generation, check if outputs are not too short
        if output_col in df.columns:
            df['output_length'] = df[output_col].astype(str).str.len()
            short_outputs = (df['output_length'] < 10).sum()
            
            if short_outputs > 0:
                issues.append({
                    "issue": "Short outputs",
                    "severity": "Warning",
                    "description": f"{short_outputs} samples have very short output text (< 10 chars), which may not be ideal for generation tasks"
                })
    
    elif task_type == 'Question Answering':
        # For QA, check if inputs look like questions
        if input_col in df.columns:
            question_marks = df[input_col].astype(str).str.contains(r'\?').sum()
            
            if question_marks / len(df) < 0.5:
                issues.append({
                    "issue": "Few questions",
                    "severity": "Warning",
                    "description": f"Only {question_marks} out of {len(df)} inputs contain question marks"
                })
    
    return len([i for i in issues if i["severity"] == "Error"]) == 0, issues 