#!/bin/bash

echo "========================================================================"
echo "macOS Threading Issue Fix"
echo "========================================================================"
echo ""
echo "This script will reinstall PyTorch and related libraries to fix"
echo "the 'mutex lock failed' error on macOS."
echo ""

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "This script is only needed on macOS."
    exit 0
fi

echo "Step 1: Uninstalling problematic packages..."
pip uninstall -y torch torchvision torchaudio

echo ""
echo "Step 2: Reinstalling PyTorch (CPU version for macOS compatibility)..."
pip install torch torchvision torchaudio

echo ""
echo "Step 3: Reinstalling transformers and sentence-transformers..."
pip install --upgrade --force-reinstall transformers sentence-transformers

echo ""
echo "Step 4: Setting up environment..."
cat > .flaskenv << 'EOF'
TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
NUMEXPR_NUM_THREADS=1
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
EOF

echo "Created .flaskenv with threading settings"

echo ""
echo "========================================================================"
echo "Fix Complete!"
echo "========================================================================"
echo ""
echo "Now try running the application:"
echo "  ./start_server.sh"
echo ""
echo "If you still see issues, try:"
echo "  conda install -c pytorch pytorch"
echo ""
