from gemma-finetuning-ui.backend.model.loader import get_available_models, load_tokenizer

# Print available models
models = get_available_models()
print("Available models:")
for model in models:
    print(f"- {model['name']}: {model['description']}")

# Test if credentials are available
from gemma-finetuning-ui.backend.model.loader import get_api_credentials
print(f"\nAPI credentials available: {'Yes' if get_api_credentials() else 'No'}")

# If credentials are available, try loading a tokenizer
if get_api_credentials():
    try:
        tokenizer = load_tokenizer("google/gemma-2b")
        print("Successfully loaded tokenizer!")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")