import gradio as gr
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Callable

def create_info_box(title: str, content: str) -> gr.Blocks:
    """Create an information box with title and content."""
    with gr.Box():
        gr.Markdown(f"### {title}")
        gr.Markdown(content)
    return gr.Box()

def create_file_upload_component(accepted_types: List[str], label: str, description: str):
    """Create a file upload component with specified parameters."""
    return gr.File(
        file_types=accepted_types,
        label=label,
        info=description
    )

def create_model_selector():
    """Create a model selection component with relevant options."""
    with gr.Group():
        gr.Markdown("### Model Selection")
        
        model_dropdown = gr.Dropdown(
            choices=["gemma-2b", "gemma-7b", "gemma-2b-it", "gemma-7b-it"],
            value="gemma-2b",
            label="Base Model",
            info="Select the Gemma model variant to fine-tune"
        )
        
        quantization = gr.Radio(
            choices=["No Quantization", "8-bit Quantization", "4-bit Quantization"],
            value="No Quantization",
            label="Model Quantization",
            info="Quantization reduces memory usage but may affect model quality"
        )
        
        device_map = gr.Radio(
            choices=["auto", "cpu", "cuda:0"],
            value="auto",
            label="Device Mapping",
            info="Where to load the model"
        )
        
        load_btn = gr.Button("Load Model", variant="primary")
        model_status = gr.Markdown("Model status: Not loaded")
        
    return model_dropdown, quantization, device_map, load_btn, model_status

def create_hyperparameter_section() -> Dict[str, Any]:
    """Create a section for hyperparameter configuration."""
    with gr.Group():
        gr.Markdown("### Training Hyperparameters")
        
        with gr.Row():
            learning_rate = gr.Slider(
                minimum=1e-6, 
                maximum=1e-3, 
                value=2e-5, 
                label="Learning Rate", 
                info="Controls how quickly the model adapts to the training data"
            )
            
            batch_size = gr.Slider(
                minimum=1, 
                maximum=64, 
                value=8, 
                step=1, 
                label="Batch Size", 
                info="Number of samples processed before model weights are updated"
            )
        
        with gr.Row():
            epochs = gr.Slider(
                minimum=1, 
                maximum=20, 
                value=3, 
                step=1, 
                label="Epochs", 
                info="Number of complete passes through the training dataset"
            )
            
            gradient_accumulation = gr.Slider(
                minimum=1, 
                maximum=16, 
                value=1, 
                step=1, 
                label="Gradient Accumulation Steps", 
                info="Accumulate gradients over multiple batches"
            )
        
        with gr.Accordion("Advanced Options", open=False):
            with gr.Row():
                weight_decay = gr.Slider(
                    minimum=0, 
                    maximum=0.1, 
                    value=0.01, 
                    label="Weight Decay", 
                    info="L2 regularization to prevent overfitting"
                )
                
                warmup_ratio = gr.Slider(
                    minimum=0, 
                    maximum=0.2, 
                    value=0.1, 
                    label="Warmup Ratio", 
                    info="Portion of training to gradually increase learning rate"
                )
            
            with gr.Row():
                optimizer = gr.Dropdown(
                    choices=["AdamW", "Adam8bit", "Lion"], 
                    value="AdamW", 
                    label="Optimizer"
                )
                
                lr_scheduler = gr.Dropdown(
                    choices=["linear", "cosine", "cosine_with_restarts", "polynomial"], 
                    value="linear", 
                    label="LR Scheduler"
                )
    
    return {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "gradient_accumulation": gradient_accumulation,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler
    }

def create_fine_tuning_method_selector():
    """Create a fine-tuning method selector component."""
    with gr.Group():
        gr.Markdown("### Fine-tuning Method")
        
        method = gr.Radio(
            choices=["Full Fine-tuning", "LoRA", "QLoRA", "Prefix Tuning"],
            value="LoRA",
            label="Method",
            info="Select which fine-tuning method to use"
        )
        
        # LoRA parameters
        with gr.Group(visible=True) as lora_params:
            gr.Markdown("#### LoRA Parameters")
            
            with gr.Row():
                lora_r = gr.Slider(1, 64, value=16, step=1, label="LoRA Rank (r)")
                lora_alpha = gr.Slider(1, 64, value=32, step=1, label="LoRA Alpha")
            
            with gr.Row():
                lora_dropout = gr.Slider(0, 0.9, value=0.1, label="LoRA Dropout")
                target_modules = gr.Dropdown(
                    choices=["all-linear", "q_proj,v_proj", "q_proj,v_proj,k_proj,o_proj", "Custom"],
                    value="q_proj,v_proj",
                    label="Target Modules",
                    multiselect=False
                )
        
        # Function to show/hide LoRA parameters based on method selection
        def update_method_visibility(method_choice):
            if method_choice in ["LoRA", "QLoRA"]:
                return gr.Group.update(visible=True)
            else:
                return gr.Group.update(visible=False)
        
        # Connect the method radio button to the visibility function
        method.change(fn=update_method_visibility, inputs=method, outputs=lora_params)
        
    return method, lora_params

def create_progress_tracker() -> Tuple[gr.Plot, gr.HTML]:
    """Create components for tracking training progress."""
    plot = gr.Plot(
        label="Training Progress", 
        show_label=True
    )
    
    status_html = gr.HTML(
        value="<div style='padding: 10px; border-radius: 5px; background-color: #f0f0f0;'>"
              "<p><b>Status:</b> Ready to start training</p>"
              "<p><b>Current step:</b> 0</p>"
              "<p><b>Time elapsed:</b> 0:00:00</p>"
              "<p><b>Estimated time remaining:</b> N/A</p>"
              "</div>"
    )
    
    return plot, status_html

def create_evaluation_metrics_display() -> gr.DataFrame:
    """Create a component to display evaluation metrics."""
    return gr.DataFrame(
        headers=["Metric", "Value"],
        datatype=["str", "number"],
        label="Evaluation Metrics"
    )

def create_model_output_tester(
    input_handler: Callable,
    output_handler: Callable
) -> Tuple[gr.Textbox, gr.Button, gr.Textbox]:
    """Create components for testing model outputs."""
    input_text = gr.Textbox(
        lines=3,
        placeholder="Enter text to test the model...",
        label="Input Text"
    )
    
    test_button = gr.Button("Generate Output")
    
    output_text = gr.Textbox(
        lines=5,
        label="Model Output",
        interactive=False
    )
    
    test_button.click(
        fn=input_handler,
        inputs=input_text,
        outputs=output_text
    )
    
    return input_text, test_button, output_text

def create_export_options() -> Dict[str, Any]:
    """Create components for model export options."""
    with gr.Group():
        gr.Markdown("### Export Options")
        
        export_format = gr.Radio(
            choices=["HuggingFace Model", "ONNX", "TensorFlow SavedModel", "PyTorch (state_dict)", "GGUF"],
            value="HuggingFace Model",
            label="Export Format"
        )
        
        with gr.Accordion("Optimization Options", open=False):
            with gr.Row():
                quantization = gr.Dropdown(
                    choices=["None", "INT8", "INT4", "FP16"],
                    value="None",
                    label="Quantization"
                )
                
                device_target = gr.Dropdown(
                    choices=["CPU", "GPU", "CPU+GPU", "Mobile"],
                    value="CPU+GPU",
                    label="Target Device"
                )
        
        export_path = gr.Textbox(
            label="Export Path",
            placeholder="/path/to/export/directory",
            value="./exported_models"
        )
        
        export_button = gr.Button("Export Model")
    
    return {
        "format": export_format,
        "quantization": quantization,
        "device_target": device_target,
        "path": export_path,
        "button": export_button
    }