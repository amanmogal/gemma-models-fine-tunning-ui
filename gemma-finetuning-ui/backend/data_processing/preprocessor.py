"""
Dataset preprocessing utilities for the Gemma Fine-tuning UI.
"""

import os
import json
import csv
import re
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datasets import Dataset, DatasetDict

def detect_file_format(file_path: str) -> str:
    """
    Detect the format of the input file based on extension and content.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        Detected format: 'csv', 'jsonl', 'txt', or 'excel'
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.csv':
        return 'csv'
    elif ext == '.jsonl' or ext == '.json':
        # Check if it's actually JSONL by reading the first line
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            try:
                json.loads(first_line)
                return 'jsonl'
            except json.JSONDecodeError:
                # If it's not valid JSON, fall back to txt
                return 'txt'
    elif ext == '.xlsx' or ext == '.xls':
        return 'excel'
    else:
        # Default to txt for unknown formats
        return 'txt'

def load_dataset(
    file_path: str,
    format_type: str = 'auto',
    delimiter: str = ',',
    has_header: bool = True,
    input_col: str = 'input',
    output_col: str = 'output'
) -> pd.DataFrame:
    """
    Load a dataset from a file.
    
    Args:
        file_path: Path to the input file
        format_type: Format of the file ('auto', 'csv', 'jsonl', 'txt', 'excel')
        delimiter: Delimiter for CSV files
        has_header: Whether CSV files have a header row
        input_col: Name of the input column
        output_col: Name of the output column
        
    Returns:
        Pandas DataFrame containing the dataset
    """
    # Auto-detect format if not specified
    if format_type == 'auto':
        format_type = detect_file_format(file_path)
    
    # Load based on format
    if format_type == 'csv':
        df = pd.read_csv(
            file_path,
            delimiter=delimiter,
            header=0 if has_header else None
        )
        
        # If no header, assign column names
        if not has_header:
            df.columns = [f'col_{i}' for i in range(len(df.columns))]
            
            # Try to guess which columns are input/output
            if len(df.columns) >= 2:
                input_col = df.columns[0]
                output_col = df.columns[1]
    
    elif format_type == 'jsonl':
        # Read JSONL line by line
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        df = pd.DataFrame(data)
        
        # Try to guess input/output columns if they don't exist
        if input_col not in df.columns and 'input' in df.columns:
            input_col = 'input'
        elif input_col not in df.columns and 'prompt' in df.columns:
            input_col = 'prompt'
        elif input_col not in df.columns and 'question' in df.columns:
            input_col = 'question'
            
        if output_col not in df.columns and 'output' in df.columns:
            output_col = 'output'
        elif output_col not in df.columns and 'response' in df.columns:
            output_col = 'response'
        elif output_col not in df.columns and 'answer' in df.columns:
            output_col = 'answer'
    
    elif format_type == 'excel':
        df = pd.read_excel(file_path, header=0 if has_header else None)
        
        # If no header, assign column names
        if not has_header:
            df.columns = [f'col_{i}' for i in range(len(df.columns))]
    
    elif format_type == 'txt':
        # For text files, assume one example per line or paragraph
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newline (paragraphs) or single newline
        if '\n\n' in content:
            examples = content.split('\n\n')
        else:
            examples = content.split('\n')
        
        # Remove empty examples
        examples = [ex.strip() for ex in examples if ex.strip()]
        
        # Create a dataframe with just input column
        df = pd.DataFrame({input_col: examples})
        
        # If there's a consistent delimiter like ":" or "###", try to split into input/output
        if ':' in content:
            # Try to split each example at the first colon
            inputs = []
            outputs = []
            
            for ex in examples:
                parts = ex.split(':', 1)
                if len(parts) == 2:
                    inputs.append(parts[0].strip())
                    outputs.append(parts[1].strip())
                else:
                    inputs.append(ex)
                    outputs.append('')
            
            df = pd.DataFrame({input_col: inputs, output_col: outputs})
    
    else:
        raise ValueError(f"Unsupported format: {format_type}")
    
    # Ensure the required columns exist
    if input_col not in df.columns:
        raise ValueError(f"Input column '{input_col}' not found in the dataset")
    
    # For datasets without output column, create an empty one
    if output_col not in df.columns:
        df[output_col] = ''
    
    return df

def clean_text(text: str) -> str:
    """
    Clean text by removing HTML tags, normalizing whitespace, etc.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def preprocess_dataset(
    df: pd.DataFrame,
    input_col: str = 'input',
    output_col: str = 'output',
    max_length: int = 512,
    truncation: bool = True,
    clean_text_flag: bool = True,
    lowercase: bool = False
) -> Tuple[Dataset, Dict[str, Any]]:
    """
    Preprocess a dataset for fine-tuning.
    
    Args:
        df: Input DataFrame
        input_col: Name of the input column
        output_col: Name of the output column
        max_length: Maximum sequence length
        truncation: Whether to truncate sequences longer than max_length
        clean_text_flag: Whether to clean the text
        lowercase: Whether to convert text to lowercase
        
    Returns:
        Tuple of (preprocessed dataset, dataset info)
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Clean text if requested
    if clean_text_flag:
        df[input_col] = df[input_col].apply(clean_text)
        df[output_col] = df[output_col].apply(clean_text)
    
    # Convert to lowercase if requested
    if lowercase:
        df[input_col] = df[input_col].str.lower()
        df[output_col] = df[output_col].str.lower()
    
    # Truncate if requested
    if truncation:
        df[input_col] = df[input_col].apply(lambda x: x[:max_length] if isinstance(x, str) else x)
        df[output_col] = df[output_col].apply(lambda x: x[:max_length] if isinstance(x, str) else x)
    
    # Calculate dataset statistics
    input_lengths = df[input_col].str.len()
    output_lengths = df[output_col].str.len()
    
    dataset_info = {
        "num_samples": len(df),
        "input_avg_length": input_lengths.mean(),
        "output_avg_length": output_lengths.mean(),
        "input_max_length": input_lengths.max(),
        "output_max_length": output_lengths.max(),
        "columns": list(df.columns)
    }
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    
    return dataset, dataset_info

def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    seed: int = 42
) -> DatasetDict:
    """
    Split a dataset into training, validation, and test sets.
    
    Args:
        dataset: Input dataset
        train_ratio: Ratio of training examples
        val_ratio: Ratio of validation examples
        test_ratio: Ratio of test examples
        shuffle: Whether to shuffle the dataset before splitting
        seed: Random seed for reproducibility
        
    Returns:
        DatasetDict containing 'train', 'validation', and 'test' splits
    """
    # Normalize ratios to sum to 1
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    test_ratio /= total
    
    # Calculate split sizes
    train_size = train_ratio
    val_size = val_ratio
    
    # Split the dataset
    splits = dataset.train_test_split(
        test_size=(val_ratio + test_ratio),
        shuffle=shuffle,
        seed=seed
    )
    
    train_test = splits['test'].train_test_split(
        test_size=test_ratio / (val_ratio + test_ratio),
        shuffle=shuffle,
        seed=seed
    )
    
    # Create the final DatasetDict
    return DatasetDict({
        'train': splits['train'],
        'validation': train_test['train'],
        'test': train_test['test']
    })

def format_for_task(
    dataset: DatasetDict,
    task_type: str,
    input_col: str = 'input',
    output_col: str = 'output'
) -> DatasetDict:
    """
    Format a dataset for a specific task type.
    
    Args:
        dataset: Input dataset
        task_type: Task type ('Text Generation', 'Classification', etc.)
        input_col: Name of the input column
        output_col: Name of the output column
        
    Returns:
        Formatted dataset
    """
    # Create a copy to avoid modifying the original
    formatted_dataset = DatasetDict()
    
    for split, ds in dataset.items():
        if task_type == 'Text Generation':
            # For text generation, format as prompt-completion pairs
            def format_generation(example):
                return {
                    'text': f"{example[input_col]}\n{example[output_col]}"
                }
            
            formatted_dataset[split] = ds.map(format_generation)
        
        elif task_type == 'Classification':
            # For classification, keep input and label columns
            def format_classification(example):
                return {
                    'text': example[input_col],
                    'label': example[output_col]
                }
            
            formatted_dataset[split] = ds.map(format_classification)
        
        elif task_type == 'Question Answering':
            # For QA, format as question-answer pairs
            def format_qa(example):
                return {
                    'question': example[input_col],
                    'answer': example[output_col]
                }
            
            formatted_dataset[split] = ds.map(format_qa)
        
        else:
            # Default: keep as is
            formatted_dataset[split] = ds
    
    return formatted_dataset 