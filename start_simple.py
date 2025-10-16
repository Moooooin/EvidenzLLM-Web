#!/usr/bin/env python
"""
Simple startup script that shows detailed progress and catches errors.
"""

import os
import sys

# Set environment variables
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

print("=" * 70)
print("EvidenzLLM Web Chat - Starting Server")
print("=" * 70)
print()

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded .env file")
except ImportError:
    print("⚠ python-dotenv not installed, using system environment")
except Exception as e:
    print(f"⚠ Could not load .env: {e}")

# Check API key
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    print("✗ GOOGLE_API_KEY not set!")
    print("  Please add your API key to .env file")
    sys.exit(1)
print(f"✓ GOOGLE_API_KEY is set ({api_key[:10]}...)")

# Check model directory
model_path = os.getenv('CLASSIFIER_PATH', './query_classifier_model')
if not os.path.exists(model_path):
    print(f"✗ Model directory not found: {model_path}")
    sys.exit(1)
print(f"✓ Model directory exists: {model_path}")

# Check Wikipedia data
wiki_path = os.getenv('WIKI_DATA_PATH', './data/wiki_texts.pkl')
if not os.path.exists(wiki_path):
    print(f"✗ Wikipedia data not found: {wiki_path}")
    print("  Run: python prepare_data.py")
    sys.exit(1)
print(f"✓ Wikipedia data exists: {wiki_path}")

print()
print("Starting Flask server...")
print("This may take 1-2 minutes for first-time model loading...")
print()

# Import and run app
try:
    import app
    print()
    print("=" * 70)
    print("Server started successfully!")
    print("Open http://localhost:5000 in your browser")
    print("=" * 70)
except KeyboardInterrupt:
    print("\n\nServer stopped by user")
    sys.exit(0)
except Exception as e:
    print()
    print("=" * 70)
    print("✗ ERROR STARTING SERVER")
    print("=" * 70)
    print(f"\nError: {e}")
    print("\nFull traceback:")
    import traceback
    traceback.print_exc()
    print()
    print("Troubleshooting:")
    print("1. Check if all dependencies are installed: pip install -r requirements.txt")
    print("2. Try the macOS fix: ./fix_macos_threading.sh")
    print("3. See TROUBLESHOOTING.md for more help")
    sys.exit(1)
