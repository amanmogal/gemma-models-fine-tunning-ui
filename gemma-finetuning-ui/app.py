import gradio as gr
import os
from frontend.pages import (
    create_home_page,
    create_dataset_page,
    create_model_config_page,
    create_training_page,
    create_evaluation_page,
    create_export_page
)
from backend.utils.config import load_config

# Load application configuration
config = load_config()

# Create Gradio Blocks app
with gr.Blocks(title="Gemma Fine-tuning UI", theme=gr.themes.Soft()) as app:
    # Application state
    state = gr.State({
        "dataset": None,
        "model_config": None,
        "training_args": None,
        "model": None,
        "evaluation_results": None
    })
    
    # Header
    with gr.Row():
        gr.Markdown("# Gemma Model Fine-tuning UI")
        gr.Markdown("A user-friendly interface for fine-tuning Google's Gemma language models")
    
    # Navigation tabs
    with gr.Tabs() as tabs:
        with gr.TabItem("🏠 Home"):
            create_home_page()
        
        with gr.TabItem("📊 Dataset Upload"):
            dataset_state = create_dataset_page(state)
        
        with gr.TabItem("⚙️ Model Configuration"):
            model_config_state = create_model_config_page(state)
        
        with gr.TabItem("🔄 Training"):
            training_state = create_training_page(state)
        
        with gr.TabItem("📈 Evaluation"):
            evaluation_state = create_evaluation_page(state)
        
        with gr.TabItem("💾 Export"):
            export_state = create_export_page(state)

# Launch the app
if __name__ == "__main__":
    app.launch(share=False) 