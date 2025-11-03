#!/usr/bin/env python3
"""
Shark Tank AI Analyzer - Application Launcher
"""
import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all requirements are installed"""
    try:
        import streamlit
        import pandas
        import plotly
        import sklearn
        print("All required packages are installed")
        return True
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_datasets():
    """Check if required datasets are present"""
    required_files = [
        "Shark Tank US dataset.csv",
        "Shark Tank India.csv", 
        "Shark Tank Australia dataset.csv",
        "shark_tank_merged.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("Missing required datasets:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all dataset files are in the project directory")
        return False
    else:
        print("All required datasets found")
        return True

def main():
    """Main launcher function"""
    print("Shark Tank AI Analyzer - Starting Application")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check datasets
    if not check_datasets():
        sys.exit(1)
    
    print("\nStarting Streamlit application...")
    print("The app will open in your default browser at http://localhost:8501")
    print("Press Ctrl+C to stop the application")
    print("=" * 50)
    
    try:
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
