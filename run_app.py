import os
import subprocess
import sys

def run_app():
    """Run the Gemma Fine-tuning UI application."""
    # Check if virtual environment exists
    venv_path = os.path.join(os.getcwd(), "venv")
    if not os.path.exists(venv_path):
        print("Virtual environment not found. Please run setup_environment.py first.")
        print("    python setup_environment.py")
        return
    
    # Determine path to python in virtual environment
    if os.name == 'nt':  # Windows
        python_path = os.path.join(venv_path, "Scripts", "python.exe")
    else:  # Unix/Mac
        python_path = os.path.join(venv_path, "bin", "python")
    
    # Set PYTHONPATH to include project directory
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    
    try:
        # First, try to import required packages to check if they're installed
        subprocess.run([python_path, "-c", "import gradio; import torch; import transformers"], 
                      env=env, check=True)
        
        # If imports succeed, run the main app
        print("Starting Gemma Fine-tuning UI...")
        app_path = os.path.join(os.getcwd(), "app.py")
        if not os.path.exists(app_path):
            print(f"Error: Could not find app.py at {app_path}")
            print("Make sure you're running this script from the project root directory.")
            return
            
        process = subprocess.run([python_path, app_path], 
                               env=env,
                               check=True)
        
    except subprocess.CalledProcessError as e:
        print("Error: Failed to run the application.")
        print("Make sure all dependencies are installed by running:")
        print("    pip install -r requirements.txt")
        print(f"Error details: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    run_app()
