#!/usr/bin/env python3
"""
Virtual Environment Setup Script for Shark Tank AI Analyzer
"""
import subprocess
import sys
import os
from pathlib import Path

def create_virtual_environment():
    """Create a virtual environment"""
    print("Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("Virtual environment created successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        return False

def get_venv_python():
    """Get the path to the virtual environment Python executable"""
    if os.name == 'nt':  # Windows
        return os.path.join("venv", "Scripts", "python.exe")
    else:  # Unix/Linux/MacOS
        return os.path.join("venv", "bin", "python")

def get_venv_pip():
    """Get the path to the virtual environment pip executable"""
    if os.name == 'nt':  # Windows
        return os.path.join("venv", "Scripts", "pip.exe")
    else:  # Unix/Linux/MacOS
        return os.path.join("venv", "bin", "pip")

def install_requirements():
    """Install requirements in the virtual environment"""
    print("Installing requirements...")
    try:
        pip_path = get_venv_pip()
        subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
        print("Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False

def create_activation_script():
    """Create activation scripts for different platforms"""
    # Windows batch file
    with open("activate_venv.bat", "w") as f:
        f.write("""@echo off
echo Activating Shark Tank AI Analyzer Virtual Environment...
call venv\\Scripts\\activate.bat
echo Virtual environment activated!
echo.
echo To run the application:
echo   python run_app.py
echo.
echo To deactivate:
echo   deactivate
""")
    
    # Unix/Linux/MacOS shell script
    with open("activate_venv.sh", "w") as f:
        f.write("""#!/bin/bash
echo "Activating Shark Tank AI Analyzer Virtual Environment..."
source venv/bin/activate
echo "Virtual environment activated!"
echo ""
echo "To run the application:"
echo "  python run_app.py"
echo ""
echo "To deactivate:"
echo "  deactivate"
""")
    
    # Make shell script executable on Unix systems
    if os.name != 'nt':
        os.chmod("activate_venv.sh", 0o755)

def main():
    """Main setup function"""
    print("Shark Tank AI Analyzer - Virtual Environment Setup")
    print("=" * 60)
    
    # Check if virtual environment already exists
    if os.path.exists("venv"):
        print("Virtual environment already exists!")
        response = input("Do you want to recreate it? (y/N): ").lower().strip()
        if response == 'y':
            print("Removing existing virtual environment...")
            import shutil
            shutil.rmtree("venv")
        else:
            print("Using existing virtual environment.")
            if install_requirements():
                print("\nSetup complete! Virtual environment is ready.")
                print("\nTo activate the virtual environment:")
                if os.name == 'nt':
                    print("  activate_venv.bat")
                else:
                    print("  source activate_venv.sh")
                print("\nTo run the application:")
                print("  python run_app.py")
            return
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Create activation scripts
    create_activation_script()
    
    print("\n" + "=" * 60)
    print("Setup complete! Virtual environment is ready.")
    print("\nTo activate the virtual environment:")
    if os.name == 'nt':
        print("  activate_venv.bat")
    else:
        print("  source activate_venv.sh")
    print("\nTo run the application:")
    print("  python run_app.py")
    print("\nTo deactivate the virtual environment:")
    print("  deactivate")

if __name__ == "__main__":
    main()
