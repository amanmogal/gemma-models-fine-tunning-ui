# Gemma models Fine-tuning UI

A user-friendly web interface for fine-tuning Google's Gemma language models, allowing users with limited technical expertise to customize models for specific use cases.

## Features

- **Dataset Management**: Upload, preprocess, and validate your training data
- **Model Configuration**: Configure Gemma models with optimized hyperparameters
- **Training Visualization**: Monitor training progress with real-time metrics
- **Model Evaluation**: Evaluate model performance with comprehensive metrics
- **Export Options**: Export fine-tuned models in various formats

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- Google API credentials (for Gemma model access)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/gemma-finetuning-ui.git
   cd gemma-finetuning-ui
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Google API credentials:
   ```
   export GOOGLE_API_KEY=your_api_key_here
   ```

### Running the Application

Start the Gradio web interface:

```
python app.py
```

The application will be available at http://localhost:7860

## Usage Guide

### 1. Dataset Upload

- Upload your dataset in CSV, JSONL, or text format
- Configure preprocessing options
- Validate and preview your processed data

### 2. Model Configuration

- Select Gemma model variant (2B or 7B)
- Configure training hyperparameters
- Choose fine-tuning approach (full fine-tuning or parameter-efficient methods)

### 3. Training

- Start and monitor training progress
- View real-time metrics and visualizations
- Save checkpoints during training

### 4. Evaluation

- Evaluate model performance on test data
- View comprehensive metrics
- Test model with custom inputs

### 5. Export

- Export your fine-tuned model in various formats
- Configure optimization options
- Deploy to your preferred platform

## Project Structure

```
gemma-finetuning-ui/
├── README.md                   # Project documentation
├── requirements.txt            # Dependencies
├── app.py                      # Main Gradio application entry point
├── frontend/                   # Frontend components
│   ├── components.py           # Reusable UI components
│   ├── pages.py                # Page layouts
│   └── utils.py                # Frontend utilities
├── backend/
│   ├── data_processing/        # Data handling modules
│   │   ├── __init__.py
│   │   ├── preprocessor.py     # Data preprocessing
│   │   ├── validation.py       # Input validation
│   │   └── augmentation.py     # Data augmentation (basic)
│   ├── model/                  # Model-related code
│   │   ├── __init__.py
│   │   ├── loader.py           # Model loading utilities
│   │   ├── trainer.py          # Training functionality
│   │   └── evaluation.py       # Model evaluation
│   └── utils/                  # Backend utilities
│       ├── __init__.py
│       ├── config.py           # Configuration handling
│       └── visualization.py    # Training metrics visualization
├── examples/                   # Example datasets
│   └── healthcare_sample.jsonl # Sample healthcare dataset
└── notebooks/                  # Development notebooks
    └── prototype.ipynb         # Prototype implementation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Acknowledgements

- Google for the Gemma language models
- Hugging Face for transformer libraries
- Gradio for the web interface framework 