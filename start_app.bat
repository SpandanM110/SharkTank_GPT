@echo off
echo Starting Shark Tank AI Analyzer...
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Creating it...
    python setup_venv.py
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if packages are installed
venv\Scripts\python.exe -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Installing required packages...
    venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Failed to install packages.
        pause
        exit /b 1
    )
)

REM Start the application
echo Starting Streamlit application...
echo The app will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the application
echo.
venv\Scripts\streamlit.exe run streamlit_app.py --server.port 8501 --server.address localhost

pause
