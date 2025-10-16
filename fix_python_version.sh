#!/bin/bash

echo "========================================================================"
echo "Python Version Fix"
echo "========================================================================"
echo ""

# Check current Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Current Python version: $PYTHON_VERSION"
echo ""

# Check if Python 3.10+ is available
if command -v python3.11 &> /dev/null; then
    echo "✓ Python 3.11 found!"
    PYTHON_CMD="python3.11"
elif command -v python3.10 &> /dev/null; then
    echo "✓ Python 3.10 found!"
    PYTHON_CMD="python3.10"
else
    echo "✗ Python 3.10+ not found"
    echo ""
    echo "Options:"
    echo "1. Install Python 3.11 via Homebrew:"
    echo "   brew install python@3.11"
    echo ""
    echo "2. Use Python 3.9 with older packages:"
    echo "   pip install -r requirements-py39.txt"
    echo ""
    exit 1
fi

echo ""
echo "Recreating venv with $PYTHON_CMD..."
echo ""

# Remove old venv
if [ -d "venv" ]; then
    echo "Removing old venv..."
    rm -rf venv
fi

# Create new venv
echo "Creating new venv..."
$PYTHON_CMD -m venv venv

echo ""
echo "✓ Venv created with $($PYTHON_CMD --version)"
echo ""
echo "Next steps:"
echo "1. source venv/bin/activate"
echo "2. pip install -r requirements.txt"
echo "3. source load_env.sh"
echo "4. python app.py"
echo ""
