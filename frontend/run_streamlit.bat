@echo off
REM Launch Streamlit Frontend for AI Hologram Medical Visualization System

echo.
echo ========================================
echo  AI Hologram Medical Visualization
echo  Streamlit Frontend Launcher
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "..\myenv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found at ..\myenv
    echo Please create it first: python -m venv ..\myenv
    exit /b 1
)

REM Activate virtual environment
call ..\myenv\Scripts\activate.bat

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Streamlit and dependencies...
    pip install -r requirements_streamlit.txt
)

REM Run Streamlit app
echo.
echo Starting Streamlit app...
echo Open your browser to http://localhost:8501
echo.

set API_URL=http://localhost:8000
streamlit run app_streamlit.py

pause
