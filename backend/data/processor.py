import pandas as pd
import json
from typing import Dict, Any, List, Tuple, Optional, Union
import os

def preprocess_dataset(
    file_path: str,
    input_col: str = "input",
    output_col: str = "output",
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    shuffle: bool = True,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Preprocess a dataset file.
    
    Args:
        file_path: Path to the dataset file
        input_col: Input column name
        output_col: Output column name
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        test_split: Proportion of data for testing
        shuffle: Whether to shuffle the dataset
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with train, validation, and test splits
    """
    # Determine file format from extension
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Load the dataset based on format
    if file_ext == ".csv":
        df = pd.read_csv(file_path)
    elif file_ext == ".jsonl":
        with open(file_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        df = pd.DataFrame(data)
    elif file_ext == ".xlsx":
        df = pd.read_excel(file_path)
    elif file_ext == ".txt":
        # Assume simple text format with input-output pairs separated by tabs
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        data = []
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                data.append({input_col: parts[0], output_col: parts[1]})
        
        df = pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # Ensure required columns exist
    if input_col not in df.columns or output_col not in df.columns:
        raise ValueError(f"Required columns not found in dataset: {input_col}, {output_col}")
    
    # Shuffle if requested
    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Calculate split indices
    total_rows = len(df)
    train_idx = int(train_split * total_rows)
    val_idx = train_idx + int(val_split * total_rows)
    
    # Create splits
    train_df = df.iloc[:train_idx]
    val_df = df.iloc[train_idx:val_idx]
    test_df = df.iloc[val_idx:]
    
    # Convert to format expected by training code
    train_data = train_df.to_dict("records")
    val_data = val_df.to_dict("records")
    test_data = test_df.to_dict("records")
    
    return {
        "train": train_data,
        "validation": val_data,
        "test": test_data,
        "column_mapping": {
            "input": input_col,
            "output": output_col
        }
    }