#!/usr/bin/env python
"""Test if imports work without threading issues."""

import os
import sys

# Set environment variables
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

print("Testing imports...")

try:
    print("1. Importing torch...")
    import torch
    print(f"   ✓ torch {torch.__version__}")
    
    print("2. Importing transformers...")
    from transformers import AutoTokenizer
    print(f"   ✓ transformers imported")
    
    print("3. Importing sentence_transformers...")
    from sentence_transformers import SentenceTransformer
    print(f"   ✓ sentence_transformers imported")
    
    print("4. Importing Flask...")
    from flask import Flask
    print(f"   ✓ Flask imported")
    
    print("\n✓ All imports successful!")
    print("The application should work now.")
    
except Exception as e:
    print(f"\n✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
