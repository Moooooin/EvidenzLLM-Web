#!/bin/bash

# EvidenzLLM Web Chat - Server Startup Script
# This script sets necessary environment variables and starts the Flask server

echo "========================================================================"
echo "EvidenzLLM Web Chat - Starting Server"
echo "========================================================================"
echo ""

# Set environment variables to prevent threading issues on macOS
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Load .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
    echo "✓ Environment variables loaded"
else
    echo "⚠ Warning: .env file not found"
    echo "  Copy .env.example to .env and configure your API key"
    exit 1
fi

# Check if GOOGLE_API_KEY is set
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "✗ Error: GOOGLE_API_KEY not set in .env file"
    echo "  Please add your Google Gemini API key to .env"
    exit 1
fi

echo "✓ GOOGLE_API_KEY is set"
echo ""

# Check if model directory exists
if [ ! -d "$CLASSIFIER_PATH" ] && [ ! -d "./query_classifier_model" ]; then
    echo "⚠ Warning: Query classifier model directory not found"
    echo "  Expected at: ${CLASSIFIER_PATH:-./query_classifier_model}"
fi

# Check if Wikipedia data exists
if [ ! -f "$WIKI_DATA_PATH" ] && [ ! -f "./data/wiki_texts.pkl" ]; then
    echo "⚠ Warning: Wikipedia data file not found"
    echo "  Run: python prepare_data.py"
fi

echo ""
echo "Starting Flask server..."
echo "Server will be available at: http://localhost:${PORT:-5000}"
echo ""
echo "Press Ctrl+C to stop the server"
echo "------------------------------------------------------------------------"
echo ""

# Start the server with Python flags to avoid threading issues
# -u: unbuffered output
# -X faulthandler: enable faulthandler for better error messages
PYTHONUNBUFFERED=1 python -u app.py
