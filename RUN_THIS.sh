#!/bin/bash

# COMPLETE FIX - Run this script to set everything up correctly

echo "========================================================================"
echo "EvidenzLLM Web Chat - Complete Setup"
echo "========================================================================"
echo ""

# Exit conda
echo "Step 1: Exiting conda..."
conda deactivate 2>/dev/null || true

# Remove old venv
if [ -d "venv" ]; then
    echo "Step 2: Removing old venv..."
    rm -rf venv
fi

# Create new venv with Python 3.11
echo "Step 3: Creating venv with Python 3.11..."
if ! command -v python3.11 &> /dev/null; then
    echo "✗ Python 3.11 not found!"
    echo "  Install it: brew install python@3.11"
    exit 1
fi

python3.11 -m venv venv

# Activate venv
echo "Step 4: Activating venv..."
source venv/bin/activate

# Verify Python version
PYTHON_VERSION=$(python --version)
echo "✓ Python version: $PYTHON_VERSION"

# Upgrade pip
echo "Step 5: Upgrading pip..."
pip install --upgrade pip -q

# Install requirements
echo "Step 6: Installing requirements (this may take a few minutes)..."
pip install -r requirements.txt -q

# Load environment
echo "Step 7: Loading environment variables..."
if [ -f "load_env.sh" ]; then
    source load_env.sh
else
    export $(cat .env | grep -v '^#' | xargs)
fi

# Verify API key
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "✗ GOOGLE_API_KEY not set!"
    echo "  Check your .env file"
    exit 1
fi

echo "✓ API key loaded"

echo ""
echo "========================================================================"
echo "✓ Setup Complete!"
echo "========================================================================"
echo ""
echo "Starting server..."
echo "This will take 1-2 minutes for first-time model loading..."
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================================================"
echo ""

# Start the app
python app.py
