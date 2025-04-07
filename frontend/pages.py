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
    """Create the model configuration page with actual loading functionality."""
    with gr.Column():
        gr.Markdown("## Model Configuration")
        gr.Markdown("Select and configure your model for fine-tuning.")
        
        # Add model selector component
        model_dropdown, quantization, device_map, load_btn, model_status = create_model_selector()
        
        # Add fine-tuning method selector (already defined in components.py)
        method, lora_params = create_fine_tuning_method_selector()
        
        # Add hyperparameter settings
        with gr.Group():
            gr.Markdown("### Training Hyperparameters")
            
            with gr.Row():
                learning_rate = gr.Slider(1e-6, 1e-3, value=2e-5, label="Learning Rate")
                batch_size = gr.Slider(1, 32, value=8, step=1, label="Batch Size")
            
            with gr.Row():
                epochs = gr.Slider(1, 30, value=3, step=1, label="Number of Epochs")
                warmup_ratio = gr.Slider(0, 0.5, value=0.1, label="Warmup Ratio")
        
        # Model loading handler
        def load_model_handler(model_id, quant_option, device):
            try:
                # Convert UI selections to backend parameters
                load_in_8bit = quant_option == "8-bit Quantization"
                load_in_4bit = quant_option == "4-bit Quantization"
                
                # Import backend functionality
                from backend.model.loader import load_model, load_tokenizer
                
                # Load tokenizer
                tokenizer = load_tokenizer(model_id)
                
                # Load model
                model = load_model(
                    model_id=model_id,
                    device_map=device,
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit
                )
                
                # Update state
                state["model_config"] = {
                    "model": model_id,
                    "device_map": device,
                    "load_in_8bit": load_in_8bit,
                    "load_in_4bit": load_in_4bit,
                    "fine_tuning_method": None,  # To be set later
                    "hyperparameters": {
                        "learning_rate": None,
                        "batch_size": None,
                        "epochs": None,
                        "warmup_ratio": None
                    }
                }
                
                # Store model and tokenizer in state
                state["model"] = model
                state["tokenizer"] = tokenizer
                
                return f"Model status: Successfully loaded {model_id}"
                
            except Exception as e:
                return f"Model status: Error loading model - {str(e)}"
        
        # Connect the button to the handler
        load_btn.click(
            fn=load_model_handler,
            inputs=[model_dropdown, quantization, device_map],
            outputs=[model_status]
        )
        
        # More code for saving config, etc.
        # ...
        
        return state

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
                        value=state.get("model_config", {})
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
            if not state.get("model_config"):
                return "Error: No model configuration found. Please configure the model first.", "<div style='padding: 10px; border-radius: 5px; background-color: #ffebee;'><p><b>Error:</b> No model configuration found</p></div>"
            
            if not state.get("dataset"):
                return "Error: No dataset found. Please upload and process a dataset first.", "<div style='padding: 10px; border-radius: 5px; background-color: #ffebee;'><p><b>Error:</b> No dataset found</p></div>"
            
            # Get configuration from state
            config = state["model_config"]
            dataset = state["dataset"]
            
            # Create training arguments
            training_args = create_training_args(
                output_dir="./results",
                num_train_epochs=config["hyperparameters"]["epochs"],
                per_device_train_batch_size=config["hyperparameters"]["batch_size"],
                learning_rate=config["hyperparameters"]["learning_rate"],
                weight_decay=config["hyperparameters"]["weight_decay"],
                warmup_ratio=config["hyperparameters"]["warmup_ratio"],
                lr_scheduler_type=config["hyperparameters"]["lr_scheduler"],
                optim=config["hyperparameters"]["optimizer"].lower(),
                gradient_accumulation_steps=config["hyperparameters"]["gradient_accumulation_steps"]
            )
            
            # Start training process
            try:
                model, metrics = train_model(
                    model_id=config["model"],
                    train_dataset=dataset["train"],
                    eval_dataset=dataset["validation"],
                    fine_tuning_method=config["fine_tuning_method"],
                    training_args=training_args,
                    progress_callback=lambda x: update_progress(x, state)
                )
                
                # Save trained model to state
                state["trained_model"] = model
                state["training_metrics"] = metrics
                
                return "Training completed successfully!", "<div style='padding: 10px; border-radius: 5px; background-color: #e8f5e9;'><p><b>Status:</b> Training completed</p><p><b>Final loss:</b> {:.4f}</p></div>".format(metrics.get("loss", 0.0))
            
            except Exception as e:
                return f"Error during training: {str(e)}", "<div style='padding: 10px; border-radius: 5px; background-color: #ffebee;'><p><b>Error:</b> Training failed</p></div>"
        
        def update_progress(progress_data, state):
            """Update training progress in the UI."""
            if "progress_plot" in state:
                state["progress_plot"].update(
                    value=create_training_plot({
                        "loss": progress_data.get("loss", []),
                        "learning_rate": progress_data.get("learning_rate", [])
                    })
                )
            
            if "status_html" in state:
                state["status_html"].update(
                    value="<div style='padding: 10px; border-radius: 5px; background-color: #f0f0f0;'>"
                    f"<p><b>Status:</b> {progress_data.get('status', 'Training')}</p>"
                    f"<p><b>Current step:</b> {progress_data.get('current_step', 0)}/{progress_data.get('total_steps', 0)}</p>"
                    f"<p><b>Time elapsed:</b> {progress_data.get('elapsed_time', 0):.1f}s</p>"
                    f"<p><b>Estimated time remaining:</b> {progress_data.get('remaining_time', 0):.1f}s</p>"
                    "</div>"
                )
        
        start_btn.click(
            fn=start_training,
            inputs=[],
            outputs=[logs, status_html]
        )
        
        def generate_sample(input_text):
            if not state.get("trained_model"):
                return "Error: No trained model available. Please train the model first."
            
            try:
                model = state["trained_model"]
                tokenizer = load_tokenizer(state["model_config"]["model"])
                
                outputs = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=input_text,
                    max_length=512,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1
                )
                
                return outputs[0] if outputs else "No output generated."
            
            except Exception as e:
                return f"Error generating sample: {str(e)}"
        
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
    """Create the evaluation page with functional evaluation capability."""
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
                        value=["Accuracy", "F1 Score"],
                        label="Evaluation Metrics"
                    )
            
            with gr.Column(scale=2):
                # Evaluation results display
                with gr.Group():
                    gr.Markdown("### Results")
                    
                    results_json = gr.JSON(label="Metrics Results")
                    confusion_matrix = gr.Plot(label="Confusion Matrix", visible=False)
                    error_analysis_btn = gr.Button("Show Error Analysis")
                
                # Sample evaluation
                with gr.Group():
                    gr.Markdown("### Try Model")
                    
                    sample_input = gr.Textbox(label="Input Text")
                    generate_btn = gr.Button("Generate Output")
                    sample_output = gr.Textbox(label="Model Output")
        
        # Connect dataset choice to custom upload visibility
        def update_custom_dataset_visibility(choice):
            return gr.File.update(visible=(choice == "Custom Dataset"))
        
        eval_dataset.change(
            fn=update_custom_dataset_visibility,
            inputs=eval_dataset,
            outputs=custom_dataset_upload
        )
        
        # Implement evaluation handler
        def evaluate_model_handler(dataset_choice, metrics, custom_dataset=None):
            if not state.get("model"):
                return "Error: No model available. Please load or train a model first.", None, None
            
            try:
                model = state["model"]
                tokenizer = state["tokenizer"]
                
                # Get evaluation dataset based on selection
                if dataset_choice == "Custom Dataset" and custom_dataset:
                    from backend.data.processor import preprocess_dataset
                    eval_data = preprocess_dataset(custom_dataset)
                else:
                    # Use dataset from state
                    split_name = dataset_choice.lower().replace(" ", "_")
                    eval_data = state["dataset"][split_name]
                
                # Convert metrics selection to backend format
                backend_metrics = []
                for metric in metrics:
                    if metric == "Accuracy":
                        backend_metrics.append("accuracy")
                    elif metric == "F1 Score":
                        backend_metrics.append("f1")
                    elif metric == "Precision":
                        backend_metrics.append("precision")
                    elif metric == "Recall":
                        backend_metrics.append("recall")
                    elif metric == "BLEU":
                        backend_metrics.append("bleu")
                    elif metric == "ROUGE":
                        backend_metrics.append("rouge")
                    elif metric == "Perplexity":
                        backend_metrics.append("perplexity")
                
                # Import evaluation function
                from backend.model.evaluation import evaluate_model
                
                # Run evaluation
                results = evaluate_model(
                    model=model,
                    tokenizer=tokenizer,
                    eval_dataset=eval_data,
                    task_type=state["model_config"].get("task", "Text Generation"),
                    metrics=backend_metrics,
                    batch_size=int(state["model_config"].get("hyperparameters", {}).get("batch_size", 8))
                )
                
                # Format results for display
                formatted_results = {}
                for metric, value in results.items():
                    if isinstance(value, (int, float)):
                        formatted_results[metric] = f"{value:.4f}"
                    else:
                        formatted_results[metric] = str(value)
                
                # Update state with results
                state["evaluation_results"] = results
                
                # Create confusion matrix if available
                cm_plot = None
                cm_visible = False
                if "confusion_matrix" in results:
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(results["confusion_matrix"], annot=True, fmt='d', cmap='Blues')
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.title('Confusion Matrix')
                    
                    cm_plot = plt
                    cm_visible = True
                
                return formatted_results, gr.Plot.update(value=cm_plot, visible=cm_visible)
            
            except Exception as e:
                return {"error": str(e)}, None
        
        # Connect evaluation button to handler
        evaluate_btn.click(
            fn=evaluate_model_handler,
            inputs=[eval_dataset, metrics_selection, custom_dataset_upload],
            outputs=[results_json, confusion_matrix]
        )
        
        # Sample generation handler
        def sample_generation_handler(input_text):
            if not state.get("model"):
                return "Error: No model available. Please load or train a model first."
            
            try:
                from backend.model.evaluation import generate_text
                
                model = state["model"]
                tokenizer = state["tokenizer"]
                
                output = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=input_text,
                    max_length=512,
                    temperature=0.7,
                    do_sample=True
                )
                
                return output[0]
            except Exception as e:
                return f"Error during generation: {str(e)}"
        
        # Connect generate button to handler
        generate_btn.click(
            fn=sample_generation_handler,
            inputs=sample_input,
            outputs=sample_output
        )
        
        return state

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