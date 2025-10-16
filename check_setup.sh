#!/bin/bash

echo "========================================================================"
echo "Setup Verification"
echo "========================================================================"
echo ""

# Check if in venv
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment active: $VIRTUAL_ENV"
else
    echo "✗ Not in virtual environment"
    echo "  Run: source venv/bin/activate"
    exit 1
fi

# Check Python location
PYTHON_PATH=$(which python)
if [[ "$PYTHON_PATH" == *"venv"* ]]; then
    echo "✓ Using venv Python: $PYTHON_PATH"
else
    echo "⚠ Python not from venv: $PYTHON_PATH"
fi

# Check if python-dotenv is installed
if python -c "import dotenv" 2>/dev/null; then
    echo "✓ python-dotenv installed"
else
    echo "✗ python-dotenv not installed"
    echo "  Run: pip install python-dotenv"
    exit 1
fi

# Check if API key is set
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "✗ GOOGLE_API_KEY not set"
    echo "  Run: source load_env.sh"
    exit 1
else
    echo "✓ GOOGLE_API_KEY is set (${GOOGLE_API_KEY:0:10}...)"
fi

# Check if .env file exists
if [ -f .env ]; then
    echo "✓ .env file exists"
else
    echo "✗ .env file not found"
    exit 1
fi

# Check if model directory exists
if [ -d "query_classifier_model" ]; then
    echo "✓ Model directory exists"
else
    echo "✗ Model directory not found"
fi

# Check if Wikipedia data exists
if [ -f "data/wiki_texts.pkl" ]; then
    echo "✓ Wikipedia data exists"
else
    echo "⚠ Wikipedia data not found (run: python prepare_data.py)"
fi

echo ""
echo "========================================================================"
echo "✓ Setup looks good! You can start the server."
echo "========================================================================"
echo ""
echo "Run: python app_minimal.py  (for testing)"
echo "Or:  python app.py          (for full AI)"
echo ""
