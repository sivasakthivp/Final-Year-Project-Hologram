#!/bin/bash
# Launch Streamlit Frontend for AI Hologram Medical Visualization System

echo ""
echo "========================================"
echo "  AI Hologram Medical Visualization"
echo "  Streamlit Frontend Launcher"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -f "../myenv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found at ../myenv"
    echo "Please create it first: python -m venv ../myenv"
    exit 1
fi

# Activate virtual environment
source ../myenv/bin/activate

# Check if streamlit is installed
python -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing Streamlit and dependencies..."
    pip install -r requirements_streamlit.txt
fi

# Run Streamlit app
echo ""
echo "Starting Streamlit app..."
echo "Open your browser to http://localhost:8501"
echo ""

export API_URL=http://localhost:8000
streamlit run app_streamlit.py
