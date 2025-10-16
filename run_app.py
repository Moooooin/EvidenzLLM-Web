#!/usr/bin/env python
"""
Wrapper script to run app.py with proper environment setup.
This ensures environment variables are set before any imports.
"""

import os
import sys

# CRITICAL: Set these BEFORE any imports
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# macOS-specific: Disable fork safety
if sys.platform == 'darwin':
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    os.environ['no_proxy'] = '*'  # Disable proxy to avoid connection issues

print("Environment variables set for macOS compatibility")
print("Starting Flask server...")

# Now run the app
exec(open('app.py').read())
