import gradio as gr
import os
import json
from typing import Dict, Any, List, Optional

from frontend.components import (
    create_info_box,
    create_file_upload_component,
    create_model_selector,
    create_hyperparameter_section,
    create_fine_tuning_method_selector,
    create_progress_tracker,
    create_evaluation_metrics_display,
    create_model_output_tester,
    create_export_options
)

from backend.data_processing.preprocessor import preprocess_dataset
from backend.data_processing.validation import validate_dataset
from backend.model.loader import get_available_models
from backend.model.trainer import train_model
from backend.model.evaluation import evaluate_model
from backend.utils.visualization import create_training_plot

def create_home_page() -> None:
    """Create the home page with project information."""
    with gr.Column():
        gr.Markdown("""
        # Welcome to Gemma Fine-tuning UI
        
        This application provides a user-friendly interface for fine-tuning Google's Gemma language models.
        
        ## What You Can Do
        
        - Upload and preprocess your dataset
        - Configure Gemma model parameters
        - Train the model with optimized settings
        - Evaluate model performance
        - Export your fine-tuned model
        
        ## Getting Started
        
        Navigate through the tabs above to start your fine-tuning journey.
        
        1. Begin by uploading your dataset in the **Dataset Upload** tab
        2. Configure your model in the **Model Configuration** tab
        3. Train your model in the **Training** tab
        4. Evaluate performance in the **Evaluation** tab
        5. Export your model in the **Export** tab
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### System Requirements")
                gr.Markdown("""
                - Python 3.8+
                - CUDA-compatible GPU (recommended)
                - 16GB+ RAM
                - 50GB+ disk space
                """)
            
            with gr.Column(scale=1):
                gr.Markdown("### Supported Models")
                gr.Markdown("""
                - Gemma 2B
                - Gemma 7B
                """)
            
            with gr.Column(scale=1):
                gr.Markdown("### Supported Tasks")
                gr.Markdown("""
                - Text classification
                - Text generation
                - Question answering
                - Summarization
                """)

def create_dataset_page(state: gr.State) -> Dict[str, Any]:
    """Create the dataset upload and preprocessing page."""
    with gr.Column():
        gr.Markdown("## Dataset Upload and Preprocessing")
        gr.Markdown("Upload your dataset and configure preprocessing options.")
        
        with gr.Row():
            with gr.Column(scale=2):
                # File upload section
                upload_file = create_file_upload_component(
                    accepted_types=[".csv", ".jsonl", ".txt", ".xlsx"],
                    label="Upload Dataset",
                    description="Upload your dataset file (CSV, JSONL, TXT, or XLSX)"
                )
                
                # Format selection
                format_selector = gr.Radio(
                    choices=["Auto-detect", "CSV", "JSONL", "Text", "Excel"],
                    value="Auto-detect",
                    label="File Format"
                )
                
                # CSV specific options (shown conditionally)
                with gr.Group(visible=False) as csv_options:
                    delimiter = gr.Textbox(
                        value=",",
                        label="Delimiter",
                        max_lines=1
                    )
                    
                    has_header = gr.Checkbox(
                        value=True,
                        label="Has Header Row"
                    )
                
                # Column mapping
                with gr.Group():
                    gr.Markdown("### Column Mapping")
                    
                    with gr.Row():
                        input_col = gr.Textbox(
                            value="input",
                            label="Input Column",
                            max_lines=1
                        )
                        
                        output_col = gr.Textbox(
                            value="output",
                            label="Output Column",
                            max_lines=1
                        )
            
            with gr.Column(scale=2):
                # Preprocessing options
                with gr.Group():
                    gr.Markdown("### Preprocessing Options")
                    
                    with gr.Row():
                        max_length = gr.Slider(
                            minimum=16,
                            maximum=2048,
                            value=512,
                            step=16,
                            label="Max Sequence Length"
                        )
                        
                        truncation = gr.Checkbox(
                            value=True,
                            label="Enable Truncation"
                        )
                    
                    with gr.Row():
                        clean_text = gr.Checkbox(
                            value=True,
                            label="Clean Text",
                            info="Remove HTML, normalize whitespace, etc."
                        )
                        
                        lowercase = gr.Checkbox(
                            value=False,
                            label="Convert to Lowercase"
                        )
                
                # Data splitting
                with gr.Group():
                    gr.Markdown("### Dataset Splitting")
                    
                    with gr.Row():
                        train_split = gr.Slider(
                            minimum=0.5,
                            maximum=0.95,
                            value=0.8,
                            label="Training Split"
                        )
                        
                        val_split = gr.Slider(
                            minimum=0.05,
                            maximum=0.3,
                            value=0.1,
                            label="Validation Split"
                        )
                        
                        test_split = gr.Slider(
                            minimum=0.0,
                            maximum=0.2,
                            value=0.1,
                            label="Test Split"
                        )
                    
                    shuffle = gr.Checkbox(
                        value=True,
                        label="Shuffle Dataset"
                    )
                    
                    seed = gr.Number(
                        value=42,
                        label="Random Seed",
                        precision=0
                    )
        
        # Preview and validation section
        with gr.Row():
            with gr.Column():
                process_btn = gr.Button("Process Dataset")
                
                dataset_info = gr.JSON(
                    label="Dataset Information",
                    value={}
                )
            
            with gr.Column():
                validation_output = gr.DataFrame(
                    headers=["Issue", "Severity", "Description"],
                    label="Validation Results"
                )
        
        # Sample preview
        preview_data = gr.DataFrame(
            label="Data Preview"
        )
        
        # Logic for format selection
        def update_format_options(format_choice):
            if format_choice == "CSV":
                return gr.Group.update(visible=True)
            else:
                return gr.Group.update(visible=False)
        
        format_selector.change(fn=update_format_options, inputs=format_selector, outputs=csv_options)
        
        # Process dataset button handler
        def handle_process_dataset(
            file, format_choice, delim, header, input_column, output_column,
            max_len, trunc, clean, lower, train, val, test, shuf, rnd_seed
        ):
            # This would call the backend processing function
            # For now, we'll return placeholder data
            dataset_info_val = {
                "num_samples": 1000,
                "input_avg_length": 128,
                "output_avg_length": 64,
                "format": format_choice,
                "columns": [input_column, output_column],
                "splits": {
                    "train": int(1000 * train),
                    "validation": int(1000 * val),
                    "test": int(1000 * test)
                }
            }
            
            validation_results = [
                ["Missing values", "Warning", "3 rows have missing values in output column"],
                ["Long sequences", "Info", "15 inputs exceed the max sequence length and will be truncated"]
            ]
            
            preview = [
                {"input": "What is machine learning?", "output": "Machine learning is a subset of artificial intelligence..."},
                {"input": "Explain fine-tuning.", "output": "Fine-tuning is the process of taking a pre-trained model..."},
                {"input": "What are transformers?", "output": "Transformers are a type of neural network architecture..."}
            ]
            
            return dataset_info_val, validation_results, preview
        
        process_btn.click(
            fn=handle_process_dataset,
            inputs=[
                upload_file, format_selector, delimiter, has_header, 
                input_col, output_col, max_length, truncation,
                clean_text, lowercase, train_split, val_split,
                test_split, shuffle, seed
            ],
            outputs=[dataset_info, validation_output, preview_data]
        )
    
    return {
        "upload_file": upload_file,
        "format": format_selector,
        "input_col": input_col,
        "output_col": output_col,
        "max_length": max_length,
        "dataset_info": dataset_info
    }

def create_model_config_page(state: gr.State) -> Dict[str, Any]:
    """Create the model configuration page."""
    with gr.Column():
        gr.Markdown("## Model Configuration")
        gr.Markdown("Select and configure your Gemma model for fine-tuning.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model selection
                available_models = ["gemma-2b", "gemma-7b"]
                model_selector = create_model_selector(
                    available_models=available_models,
                    label="Select Gemma Model"
                )
                
                # Task type
                task_type = gr.Radio(
                    choices=["Text Generation", "Classification", "Question Answering", "Summarization"],
                    value="Text Generation",
                    label="Task Type"
                )
                
                # Fine-tuning method
                method, lora_params = create_fine_tuning_method_selector()
            
            with gr.Column(scale=2):
                # Hyperparameters
                hyperparams = create_hyperparameter_section()
        
        # Hardware configuration
        with gr.Group():
            gr.Markdown("### Hardware Configuration")
            
            with gr.Row():
                device = gr.Radio(
                    choices=["CPU", "GPU", "Mixed Precision"],
                    value="Mixed Precision",
                    label="Compute Device"
                )
                
                precision = gr.Dropdown(
                    choices=["fp32", "fp16", "bf16"],
                    value="bf16",
                    label="Precision",
                    interactive=True
                )
                
                gradient_checkpointing = gr.Checkbox(
                    value=True,
                    label="Gradient Checkpointing",
                    info="Trades compute for memory efficiency"
                )
        
        # Save configuration button
        save_config_btn = gr.Button("Save Configuration")
        
        config_status = gr.Markdown("Configuration not saved")
        
        # Save configuration handler
        def save_model_config(
            model, task, ft_method, lr, batch_size, epochs, grad_accum,
            weight_decay, warmup, optimizer, lr_sched, device_choice, prec
        ):
            # This would save the configuration to the state
            config = {
                "model": model,
                "task": task,
                "fine_tuning_method": ft_method,
                "hyperparameters": {
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "gradient_accumulation_steps": grad_accum,
                    "weight_decay": weight_decay,
                    "warmup_ratio": warmup,
                    "optimizer": optimizer,
                    "lr_scheduler": lr_sched
                },
                "hardware": {
                    "device": device_choice,
                    "precision": prec
                }
            }
            
            return "✅ Configuration saved successfully!"
        
        save_config_btn.click(
            fn=save_model_config,
            inputs=[
                model_selector, task_type, method,
                hyperparams["learning_rate"], hyperparams["batch_size"],
                hyperparams["epochs"], hyperparams["gradient_accumulation"],
                hyperparams["weight_decay"], hyperparams["warmup_ratio"],
                hyperparams["optimizer"], hyperparams["lr_scheduler"],
                device, precision
            ],
            outputs=config_status
        )
    
    return {
        "model": model_selector,
        "task": task_type,
        "method": method,
        "hyperparams": hyperparams
    }

def create_training_page(state: gr.State) -> Dict[str, Any]:
    """Create the training page."""
    with gr.Column():
        gr.Markdown("## Model Training")
        gr.Markdown("Train your configured Gemma model on the uploaded dataset.")
        
        # Training status and controls
        with gr.Row():
            with gr.Column(scale=1):
                start_btn = gr.Button("Start Training", variant="primary")
                stop_btn = gr.Button("Stop Training")
                
                with gr.Group():
                    gr.Markdown("### Training Configuration")
                    
                    config_display = gr.JSON(
                        label="Current Configuration",
                        value={
                            "model": "gemma-2b",
                            "fine_tuning_method": "LoRA",
                            "batch_size": 8,
                            "learning_rate": 2e-5,
                            "epochs": 3
                        }
                    )
            
            with gr.Column(scale=2):
                # Progress tracking
                progress_plot, status_html = create_progress_tracker()
        
        # Training logs
        with gr.Accordion("Training Logs", open=False):
            logs = gr.Textbox(
                label="Logs",
                lines=10,
                max_lines=20,
                interactive=False
            )
        
        # Checkpoint management
        with gr.Group():
            gr.Markdown("### Checkpoints")
            
            with gr.Row():
                save_checkpoint_btn = gr.Button("Save Checkpoint")
                
                checkpoint_dropdown = gr.Dropdown(
                    choices=["No checkpoints available"],
                    label="Available Checkpoints",
                    interactive=True
                )
                
                load_checkpoint_btn = gr.Button("Load Selected Checkpoint")
        
        # Sample generation during training
        with gr.Group():
            gr.Markdown("### Sample Generation")
            
            sample_input = gr.Textbox(
                lines=2,
                placeholder="Enter text to test the current model state...",
                label="Sample Input"
            )
            
            generate_btn = gr.Button("Generate Sample")
            
            sample_output = gr.Textbox(
                lines=4,
                label="Model Output",
                interactive=False
            )
        
        # Training handlers
        def start_training():
            # This would start the training process
            return "Training started...", "<div style='padding: 10px; border-radius: 5px; background-color: #f0f0f0;'><p><b>Status:</b> Training in progress</p><p><b>Current step:</b> 1/1000</p><p><b>Time elapsed:</b> 0:00:05</p><p><b>Estimated time remaining:</b> 0:50:00</p></div>"
        
        start_btn.click(
            fn=start_training,
            inputs=[],
            outputs=[logs, status_html]
        )
        
        def generate_sample(input_text):
            # This would generate a sample from the current model state
            return f"Sample output for: {input_text}\n\nThis is a generated response from the model that would show how the current training is affecting the model's outputs."
        
        generate_btn.click(
            fn=generate_sample,
            inputs=[sample_input],
            outputs=[sample_output]
        )
    
    return {
        "start_button": start_btn,
        "stop_button": stop_btn,
        "progress_plot": progress_plot,
        "status": status_html,
        "logs": logs
    }

def create_evaluation_page(state: gr.State) -> Dict[str, Any]:
    """Create the evaluation page."""
    with gr.Column():
        gr.Markdown("## Model Evaluation")
        gr.Markdown("Evaluate your fine-tuned model's performance.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Evaluation controls
                evaluate_btn = gr.Button("Evaluate Model", variant="primary")
                
                with gr.Group():
                    gr.Markdown("### Evaluation Dataset")
                    
                    eval_dataset = gr.Radio(
                        choices=["Test Split", "Validation Split", "Custom Dataset"],
                        value="Test Split",
                        label="Evaluation Dataset"
                    )
                    
                    custom_dataset_upload = gr.File(
                        label="Upload Custom Dataset",
                        file_types=[".csv", ".jsonl"],
                        visible=False
                    )
                
                # Evaluation metrics selection
                with gr.Group():
                    gr.Markdown("### Metrics")
                    
                    metrics_selection = gr.CheckboxGroup(
                        choices=["Accuracy", "F1 Score", "Precision", "Recall", "BLEU", "ROUGE", "Perplexity"],
                        value=["Accuracy", "F1 Score", "Perplexity"],
                        label="Select Metrics"
                    )
            
            with gr.Column(scale=2):
                # Results display
                metrics_display = create_evaluation_metrics_display()
                
                # Confusion matrix for classification
                with gr.Group(visible=False) as confusion_matrix_group:
                    gr.Markdown("### Confusion Matrix")
                    confusion_matrix = gr.Plot(label="Confusion Matrix")
        
        # Error analysis
        with gr.Accordion("Error Analysis", open=False):
            error_examples = gr.DataFrame(
                headers=["Input", "Expected Output", "Model Output", "Error Type"],
                label="Misclassified Examples"
            )
        
        # Interactive testing
        with gr.Group():
            gr.Markdown("### Interactive Testing")
            
            def test_model_input(text):
                # This would run inference on the model
                return f"Model output for: {text}\n\nThis is a sample response that would be generated by your fine-tuned model."
            
            def test_model_output(output):
                # Process the output if needed
                return output
            
            input_text, test_button, output_text = create_model_output_tester(
                input_handler=test_model_input,
                output_handler=test_model_output
            )
        
        # Logic for evaluation dataset selection
        def update_custom_dataset_visibility(dataset_choice):
            if dataset_choice == "Custom Dataset":
                return gr.File.update(visible=True)
            else:
                return gr.File.update(visible=False)
        
        eval_dataset.change(
            fn=update_custom_dataset_visibility,
            inputs=eval_dataset,
            outputs=custom_dataset_upload
        )
        
        # Evaluation button handler
        def run_evaluation(dataset_choice, metrics):
            # This would run the evaluation process
            # For now, return placeholder data
            metrics_data = [
                ["Accuracy", 0.92],
                ["F1 Score", 0.89],
                ["Precision", 0.91],
                ["Recall", 0.87],
                ["Perplexity", 3.45]
            ]
            
            error_data = [
                ["What is the capital of France?", "Paris", "Lyon", "Incorrect Answer"],
                ["Summarize this article.", "The article discusses climate change...", "The article is about global warming...", "Partial Match"],
                ["Translate to French.", "Bonjour, comment ça va?", "Bonjour, comment allez-vous?", "Acceptable Variation"]
            ]
            
            return metrics_data, error_data
        
        evaluate_btn.click(
            fn=run_evaluation,
            inputs=[eval_dataset, metrics_selection],
            outputs=[metrics_display, error_examples]
        )
    
    return {
        "evaluate_button": evaluate_btn,
        "metrics_display": metrics_display,
        "error_examples": error_examples
    }

def create_export_page(state: gr.State) -> Dict[str, Any]:
    """Create the export page."""
    with gr.Column():
        gr.Markdown("## Model Export")
        gr.Markdown("Export your fine-tuned model for deployment.")
        
        # Export options
        export_components = create_export_options()
        
        # Model card generation
        with gr.Group():
            gr.Markdown("### Model Card")
            
            with gr.Row():
                model_name = gr.Textbox(
                    label="Model Name",
                    placeholder="my-fine-tuned-gemma",
                    value="gemma-finetuned"
                )
                
                model_version = gr.Textbox(
                    label="Version",
                    placeholder="1.0.0",
                    value="1.0.0"
                )
            
            model_description = gr.Textbox(
                label="Model Description",
                placeholder="Describe your fine-tuned model...",
                lines=3,
                value="A fine-tuned version of Gemma for [task description]."
            )
            
            generate_card_btn = gr.Button("Generate Model Card")
            
            model_card_preview = gr.Markdown("Model card will appear here...")
        
        # Export status
        export_status = gr.Markdown("Ready to export")
        
        # Export button handler
        def handle_export(
            export_format, quantization, device_target, export_path,
            model_name_val, model_version_val, model_description_val
        ):
            # This would handle the export process
            # For now, return a status message
            return f"✅ Model successfully exported as {export_format} to {export_path}/{model_name_val}-{model_version_val}"
        
        export_components["button"].click(
            fn=handle_export,
            inputs=[
                export_components["format"],
                export_components["quantization"],
                export_components["device_target"],
                export_components["path"],
                model_name,
                model_version,
                model_description
            ],
            outputs=export_status
        )
        
        # Generate model card handler
        def generate_model_card(name, version, description):
            # This would generate a model card markdown
            return f"""
            # {name} v{version}
            
            {description}
            
            ## Model Details
            
            - **Base Model:** Gemma 2B
            - **Fine-tuning Method:** LoRA
            - **Training Dataset:** Custom dataset with 1000 examples
            - **Performance Metrics:**
              - Accuracy: 92%
              - F1 Score: 89%
              - Perplexity: 3.45
            
            ## Intended Use
            
            This model is designed for [specific use case].
            
            ## Limitations
            
            The model may not perform well on [specific limitations].
            
            ## Training Procedure
            
            - **Training Framework:** Hugging Face Transformers
            - **Optimizer:** AdamW
            - **Learning Rate:** 2e-5
            - **Epochs:** 3
            - **Batch Size:** 8
            
            ## Ethical Considerations
            
            [Ethical considerations for model usage]
            """
        
        generate_card_btn.click(
            fn=generate_model_card,
            inputs=[model_name, model_version, model_description],
            outputs=model_card_preview
        )
    
    return {
        "export_components": export_components,
        "model_name": model_name,
        "model_version": model_version,
        "export_status": export_status
    } 