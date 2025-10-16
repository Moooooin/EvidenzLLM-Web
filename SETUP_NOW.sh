#!/bin/bash

echo "========================================================================"
echo "Setting Up EvidenzLLM with Python 3.11"
echo "========================================================================"
echo ""

# Deactivate any active environment
echo "Step 1: Deactivating environments..."
deactivate 2>/dev/null || true
conda deactivate 2>/dev/null || true

# Remove old venv
echo "Step 2: Removing old venv..."
rm -rf venv

# Create new venv with Homebrew Python 3.11
echo "Step 3: Creating new venv with Python 3.11..."
/opt/homebrew/bin/python3.11 -m venv venv

# Activate venv
echo "Step 4: Activating venv..."
source venv/bin/activate

# Verify Python version
echo "Step 5: Verifying Python version..."
python --version

# Upgrade pip
echo "Step 6: Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Step 7: Installing requirements..."
echo "This will take a few minutes..."
pip install -r requirements.txt

# Download NLTK data
echo "Step 8: Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

# Load environment
echo "Step 9: Loading environment variables..."
source load_env.sh

echo ""
echo "========================================================================"
echo "✓ Setup Complete!"
echo "========================================================================"
echo ""
echo "Your venv is now active with Python 3.11"
echo ""
echo "To start the server:"
echo "  python app.py"
echo ""
echo "Or to test minimal mode:"
echo "  python app_minimal.py"
echo ""
