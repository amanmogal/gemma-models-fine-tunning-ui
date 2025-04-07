"""
Dataset augmentation utilities for the Gemma Fine-tuning UI.
"""

import random
import re
from typing import List, Dict, Any, Optional, Union
import pandas as pd
from datasets import Dataset

def augment_dataset(
    dataset: Dataset,
    input_col: str = 'input',
    output_col: str = 'output',
    augmentation_methods: List[str] = ['synonym_replacement'],
    augmentation_factor: float = 0.5,
    max_augmented_samples: Optional[int] = None,
    seed: int = 42
) -> Dataset:
    """
    Augment a dataset with various text augmentation techniques.
    
    Args:
        dataset: Dataset to augment
        input_col: Name of the input column
        output_col: Name of the output column
        augmentation_methods: List of augmentation methods to apply
        augmentation_factor: Fraction of original dataset to augment
        max_augmented_samples: Maximum number of augmented samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        Augmented dataset
    """
    random.seed(seed)
    
    # Convert to pandas for easier manipulation
    df = dataset.to_pandas()
    
    # Calculate number of samples to augment
    num_samples = len(df)
    num_to_augment = min(
        int(num_samples * augmentation_factor),
        max_augmented_samples if max_augmented_samples is not None else float('inf')
    )
    
    # Select random samples to augment
    indices_to_augment = random.sample(range(num_samples), min(num_to_augment, num_samples))
    samples_to_augment = df.iloc[indices_to_augment].copy()
    
    # Apply augmentation methods
    augmented_samples = []
    
    for _, row in samples_to_augment.iterrows():
        input_text = row[input_col]
        output_text = row[output_col] if output_col in row else None
        
        for method in augmentation_methods:
            if method == 'synonym_replacement':
                augmented_input = synonym_replacement(input_text)
                augmented_row = row.copy()
                augmented_row[input_col] = augmented_input
                augmented_samples.append(augmented_row)
            
            elif method == 'random_insertion':
                augmented_input = random_insertion(input_text)
                augmented_row = row.copy()
                augmented_row[input_col] = augmented_input
                augmented_samples.append(augmented_row)
            
            elif method == 'random_deletion':
                augmented_input = random_deletion(input_text)
                augmented_row = row.copy()
                augmented_row[input_col] = augmented_input
                augmented_samples.append(augmented_row)
            
            elif method == 'random_swap':
                augmented_input = random_swap(input_text)
                augmented_row = row.copy()
                augmented_row[input_col] = augmented_input
                augmented_samples.append(augmented_row)
    
    # Create augmented dataframe
    augmented_df = pd.DataFrame(augmented_samples)
    
    # Combine original and augmented data
    combined_df = pd.concat([df, augmented_df], ignore_index=True)
    
    # Convert back to Dataset
    return Dataset.from_pandas(combined_df)

# Simple synonym dictionary for demonstration purposes
# In a real implementation, this would use a more comprehensive synonym database
SYNONYMS = {
    'good': ['great', 'excellent', 'fine', 'positive'],
    'bad': ['poor', 'terrible', 'awful', 'negative'],
    'happy': ['glad', 'pleased', 'delighted', 'joyful'],
    'sad': ['unhappy', 'depressed', 'gloomy', 'miserable'],
    'big': ['large', 'huge', 'enormous', 'gigantic'],
    'small': ['tiny', 'little', 'miniature', 'compact'],
    'important': ['significant', 'crucial', 'essential', 'vital'],
    'difficult': ['hard', 'challenging', 'tough', 'demanding'],
    'easy': ['simple', 'straightforward', 'effortless', 'uncomplicated'],
    'beautiful': ['attractive', 'pretty', 'lovely', 'gorgeous']
}

def synonym_replacement(text: str, n: int = 1) -> str:
    """
    Replace n words in the text with their synonyms.
    
    Args:
        text: Input text
        n: Number of words to replace
        
    Returns:
        Augmented text with synonyms
    """
    words = text.split()
    if len(words) == 0:
        return text
    
    # Find words that have synonyms
    candidates = []
    for i, word in enumerate(words):
        word_lower = word.lower().strip('.,!?;:')
        if word_lower in SYNONYMS:
            candidates.append((i, word, word_lower))
    
    # If no candidates found, return original text
    if not candidates:
        return text
    
    # Replace up to n words
    num_to_replace = min(n, len(candidates))
    indices_to_replace = random.sample(range(len(candidates)), num_to_replace)
    
    for idx in indices_to_replace:
        i, original_word, word_lower = candidates[idx]
        synonym = random.choice(SYNONYMS[word_lower])
        
        # Preserve capitalization
        if original_word[0].isupper():
            synonym = synonym.capitalize()
        
        # Preserve punctuation
        if not original_word[-1].isalnum():
            synonym += original_word[-1]
        
        words[i] = synonym
    
    return ' '.join(words)

def random_insertion(text: str, n: int = 1) -> str:
    """
    Randomly insert n words into the text.
    
    Args:
        text: Input text
        n: Number of words to insert
        
    Returns:
        Augmented text with insertions
    """
    words = text.split()
    if len(words) == 0:
        return text
    
    # Get all words that have synonyms
    all_synonyms = []
    for word in words:
        word_lower = word.lower().strip('.,!?;:')
        if word_lower in SYNONYMS:
            all_synonyms.extend(SYNONYMS[word_lower])
    
    # If no synonyms found, return original text
    if not all_synonyms:
        return text
    
    # Insert up to n words
    for _ in range(n):
        # Choose a random position to insert
        insert_pos = random.randint(0, len(words))
        
        # Choose a random word to insert
        insert_word = random.choice(all_synonyms)
        
        words.insert(insert_pos, insert_word)
    
    return ' '.join(words)

def random_deletion(text: str, p: float = 0.1) -> str:
    """
    Randomly delete words from the text with probability p.
    
    Args:
        text: Input text
        p: Probability of deleting each word
        
    Returns:
        Augmented text with deletions
    """
    words = text.split()
    if len(words) <= 1:
        return text
    
    # Randomly delete words
    new_words = []
    for word in words:
        if random.random() >= p:
            new_words.append(word)
    
    # If all words were deleted, keep one random word
    if len(new_words) == 0:
        return random.choice(words)
    
    return ' '.join(new_words)

def random_swap(text: str, n: int = 1) -> str:
    """
    Randomly swap the positions of n pairs of words in the text.
    
    Args:
        text: Input text
        n: Number of pairs to swap
        
    Returns:
        Augmented text with swapped words
    """
    words = text.split()
    if len(words) <= 1:
        return text
    
    # Swap up to n pairs
    num_swaps = min(n, len(words) // 2)
    
    for _ in range(num_swaps):
        # Choose two random positions to swap
        idx1, idx2 = random.sample(range(len(words)), 2)
        
        # Swap the words
        words[idx1], words[idx2] = words[idx2], words[idx1]
    
    return ' '.join(words) 