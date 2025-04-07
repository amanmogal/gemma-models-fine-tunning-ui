"""
Visualization utilities for the Gemma Fine-tuning UI.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import io
import base64

def create_training_plot(
    metrics: Dict[str, List[float]],
    steps: Optional[List[int]] = None,
    title: str = "Training Progress",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create a plot of training metrics.
    
    Args:
        metrics: Dictionary of metric names to lists of values
        steps: List of step numbers (x-axis)
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create steps if not provided
    if steps is None:
        steps = list(range(1, len(next(iter(metrics.values()))) + 1))
    
    # Plot each metric
    for metric_name, values in metrics.items():
        ax.plot(steps, values, label=metric_name)
    
    # Add labels and legend
    ax.set_xlabel("Steps")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    
    # Tight layout
    fig.tight_layout()
    
    return fig

def create_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues"
) -> plt.Figure:
    """
    Create a confusion matrix plot.
    
    Args:
        confusion_matrix: Confusion matrix as a numpy array
        class_names: List of class names
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create class names if not provided
    if class_names is None:
        class_names = [str(i) for i in range(confusion_matrix.shape[0])]
    
    # Plot confusion matrix
    im = ax.imshow(confusion_matrix, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Add labels
    ax.set(
        xticks=np.arange(confusion_matrix.shape[1]),
        yticks=np.arange(confusion_matrix.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label"
    )
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.0
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(
                j, i, format(confusion_matrix[i, j], "d"),
                ha="center", va="center",
                color="white" if confusion_matrix[i, j] > thresh else "black"
            )
    
    # Tight layout
    fig.tight_layout()
    
    return fig

def create_class_distribution_plot(
    class_counts: Dict[str, int],
    title: str = "Class Distribution",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create a bar plot of class distribution.
    
    Args:
        class_counts: Dictionary of class names to counts
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort classes by count
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    classes, counts = zip(*sorted_classes)
    
    # Create bar plot
    bars = ax.bar(classes, counts)
    
    # Add labels and title
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.1,
            str(int(height)),
            ha="center", va="bottom"
        )
    
    # Rotate x-axis labels if there are many classes
    if len(classes) > 5:
        plt.xticks(rotation=45, ha="right")
    
    # Tight layout
    fig.tight_layout()
    
    return fig

def create_sequence_length_histogram(
    lengths: List[int],
    title: str = "Sequence Length Distribution",
    figsize: Tuple[int, int] = (10, 6),
    bins: int = 20
) -> plt.Figure:
    """
    Create a histogram of sequence lengths.
    
    Args:
        lengths: List of sequence lengths
        title: Plot title
        figsize: Figure size
        bins: Number of histogram bins
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create histogram
    n, bins, patches = ax.hist(lengths, bins=bins, alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)
    
    # Add statistics
    stats_text = (
        f"Mean: {np.mean(lengths):.1f}\n"
        f"Median: {np.median(lengths):.1f}\n"
        f"Min: {np.min(lengths)}\n"
        f"Max: {np.max(lengths)}"
    )
    ax.text(
        0.95, 0.95, stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )
    
    # Tight layout
    fig.tight_layout()
    
    return fig

def fig_to_base64(fig: plt.Figure) -> str:
    """
    Convert a matplotlib figure to a base64-encoded string.
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        Base64-encoded string
    """
    # Save figure to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    
    # Encode as base64
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    
    # Close the figure to free memory
    plt.close(fig)
    
    return img_str

def create_learning_rate_schedule_plot(
    learning_rates: List[float],
    steps: Optional[List[int]] = None,
    title: str = "Learning Rate Schedule",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create a plot of the learning rate schedule.
    
    Args:
        learning_rates: List of learning rates
        steps: List of step numbers (x-axis)
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create steps if not provided
    if steps is None:
        steps = list(range(1, len(learning_rates) + 1))
    
    # Plot learning rate schedule
    ax.plot(steps, learning_rates)
    
    # Add labels and title
    ax.set_xlabel("Steps")
    ax.set_ylabel("Learning Rate")
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)
    
    # Use log scale for y-axis
    ax.set_yscale("log")
    
    # Tight layout
    fig.tight_layout()
    
    return fig

def create_confusion_matrix_plot(
    confusion_matrix: List[List[int]],
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Create a confusion matrix plot.
    
    Args:
        confusion_matrix: 2D list of confusion matrix values
        labels: List of class labels
        title: Plot title
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    
    # Add colorbar
    plt.colorbar(im)
    
    # Add labels
    if labels is None:
        labels = [f"Class {i}" for i in range(len(confusion_matrix))]
    
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[0])):
            text = ax.text(j, i, confusion_matrix[i][j],
                          ha="center", va="center", color="black")
    
    # Add labels and title
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    
    # Tight layout
    fig.tight_layout()
    
    return fig 