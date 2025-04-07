import os
import subprocess
import sys

def setup_environment():
    """Set up the environment for the Gemma Fine-tuning UI."""
    print("Setting up environment for Gemma Fine-tuning UI...")
    
    # Create virtual environment
    print("\n1. Creating virtual environment...")
    venv_path = os.path.join(os.getcwd(), "venv")
    if not os.path.exists(venv_path):
        subprocess.run([sys.executable, "-m", "venv", venv_path])
        print(f"Virtual environment created at: {venv_path}")
    else:
        print(f"Virtual environment already exists at: {venv_path}")
    
    # Determine path to pip in virtual environment
    if os.name == 'nt':  # Windows
        pip_path = os.path.join(venv_path, "Scripts", "pip")
    else:  # Unix/Mac
        pip_path = os.path.join(venv_path, "bin", "pip")
    
    # Upgrade pip
    print("\n2. Upgrading pip...")
    subprocess.run([pip_path, "install", "--upgrade", "pip"])
    
    # Install dependencies
    print("\n3. Installing dependencies...")
    project_dir = os.getcwd()
    requirements_path = os.path.join(project_dir, "requirements.txt")
    subprocess.run([pip_path, "install", "-r", requirements_path])
    
    # Print activation instructions
    print("\n4. Environment setup complete!")
    if os.name == 'nt':  # Windows
        activate_cmd = os.path.join(venv_path, "Scripts", "activate")
        print(f"\nTo activate the virtual environment, run:")
        print(f"    {activate_cmd}")
    else:  # Unix/Mac
        activate_cmd = os.path.join(venv_path, "bin", "activate")
        print(f"\nTo activate the virtual environment, run:")
        print(f"    source {activate_cmd}")
    
    print("\nAfter activation, run the application with:")
    print("    python app.py")

if __name__ == "__main__":
    setup_environment()
