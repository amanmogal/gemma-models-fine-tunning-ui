import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

# Configure page
title = "Gemma Model Interactive Demo"
description = """
# 🔮 Gemma Model Interactive Demo
This demo allows you to interact with Google's Gemma model for text generation tasks.
Try out different prompts to see how the model responds!
"""

# Set up model loading with error handling
def load_model():
    try:
        # Ensure you have the necessary access token for Gemma model (e.g., via HF_TOKEN environment variable)
        model_id = "google/gemma-2b"  # Can be changed to "google/gemma-7b" for the larger model
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None

# Generate text with the model
def generate_text(prompt, max_length=256, temperature=0.7, top_p=0.95):
    if not tokenizer or not model:
        return "⚠️ Model failed to load. Please check your access permissions and try again."
    
    try:
        # Format the prompt properly for Gemma
        formatted_prompt = f"{prompt}"
        
        # Generate text
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Generate with specified parameters
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )
        
        # Decode and return the response (excluding the prompt)
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        model_response = full_response[len(formatted_prompt):]
        
        return model_response
    except Exception as e:
        return f"Error generating text: {str(e)}"

# Load model and tokenizer
print("Loading Gemma model... This may take a moment.")
tokenizer, model = load_model()

# Define interface
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown(description)
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Enter your prompt",
                placeholder="Ask Gemma something or provide a starting text...",
                lines=4
            )
            
            with gr.Row():
                with gr.Column():
                    max_length = gr.Slider(
                        minimum=16, 
                        maximum=512, 
                        value=256, 
                        step=8, 
                        label="Maximum Output Length"
                    )
                with gr.Column():
                    temperature = gr.Slider(
                        minimum=0.1, 
                        maximum=1.0, 
                        value=0.7, 
                        step=0.1, 
                        label="Temperature (Creativity)"
                    )
                    top_p = gr.Slider(
                        minimum=0.5, 
                        maximum=1.0, 
                        value=0.95, 
                        step=0.05, 
                        label="Top-p (Diversity)"
                    )
            
            submit_btn = gr.Button("Generate")
            
            # Example prompts
            gr.Examples(
                [
                    ["Write a short story about a robot learning to paint."],
                    ["Explain quantum computing to a 10-year-old."],
                    ["What are three ways to improve productivity when working from home?"],
                    ["Write a function in Python that finds the prime numbers up to n."],
                ],
                prompt
            )
            
        with gr.Column():
            output = gr.Textbox(
                label="Gemma's Response",
                placeholder="Generated text will appear here...",
                lines=12
            )
    
    # Additional information and explanation
    with gr.Accordion("About Gemma", open=False):
        gr.Markdown("""
        ## About the Gemma Model
        
        Gemma is a family of lightweight, state-of-the-art open models from Google based on Gemini research and technology.
        
        The models come in two sizes:
        - Gemma 2B (used in this demo)
        - Gemma 7B (larger, more capable)
        
        For more information, visit the [Gemma documentation](https://ai.google.dev/gemma).
        
        ## Parameter Explanation
        
        - **Temperature**: Controls randomness. Lower values make responses more deterministic and focused, higher values make output more creative and diverse.
        - **Top-p (nucleus sampling)**: Controls diversity by limiting to the most likely tokens whose probabilities add up to top_p.
        - **Maximum Length**: The maximum number of tokens to generate in the response.
        """)
    
    # Set up event handlers
    submit_btn.click(
        generate_text,
        inputs=[prompt, max_length, temperature, top_p],
        outputs=output
    )

demo.launch()
