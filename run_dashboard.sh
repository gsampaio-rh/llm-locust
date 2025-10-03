#!/bin/bash
# Quick launch script for the Streamlit dashboard

echo "🎯 Starting LLM Benchmark Dashboard..."
echo ""

# Check if in venv
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Consider activating venv first: source venv/bin/activate"
    echo ""
fi

# Install/update dependencies if needed
if ! command -v streamlit &> /dev/null; then
    echo "📦 Installing Streamlit dependencies..."
    pip install -r streamlit_app/requirements.txt
    echo ""
fi

# Launch the app
cd streamlit_app
streamlit run app.py

