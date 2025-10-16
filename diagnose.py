#!/usr/bin/env python
"""
Diagnostic script to check if everything is set up correctly.
"""

import os
import sys

print("=" * 70)
print("EvidenzLLM Web Chat - System Diagnostics")
print("=" * 70)
print()

issues = []
warnings = []

# Check Python version
print("1. Checking Python version...")
version = sys.version_info
if version.major >= 3 and version.minor >= 8:
    print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
else:
    print(f"   ✗ Python {version.major}.{version.minor}.{version.micro} (need 3.8+)")
    issues.append("Python version too old")

# Check dependencies
print("\n2. Checking dependencies...")
required_packages = [
    'flask',
    'flask_cors',
    'torch',
    'transformers',
    'sentence_transformers',
    'google.generativeai',
    'nltk',
    'numpy',
    'faiss',
    'rank_bm25'
]

for package in required_packages:
    try:
        __import__(package.replace('-', '_'))
        print(f"   ✓ {package}")
    except ImportError:
        print(f"   ✗ {package} not installed")
        issues.append(f"Missing package: {package}")

# Check .env file
print("\n3. Checking .env file...")
if os.path.exists('.env'):
    print("   ✓ .env file exists")
    
    # Check API key
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except:
        pass
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key:
        print(f"   ✓ GOOGLE_API_KEY is set ({api_key[:10]}...)")
    else:
        print("   ✗ GOOGLE_API_KEY not set in .env")
        issues.append("GOOGLE_API_KEY not configured")
else:
    print("   ✗ .env file not found")
    issues.append(".env file missing")
    print("     Run: cp .env.example .env")

# Check model directory
print("\n4. Checking model directory...")
model_path = os.getenv('CLASSIFIER_PATH', './query_classifier_model')
if os.path.exists(model_path):
    print(f"   ✓ Model directory exists: {model_path}")
    
    # Check for model files
    model_files = os.listdir(model_path)
    has_safetensors = 'model.safetensors' in model_files
    has_pytorch = 'pytorch_model.bin' in model_files
    
    if has_safetensors or has_pytorch:
        format_type = 'safetensors' if has_safetensors else 'pytorch_model.bin'
        print(f"   ✓ Model file found: {format_type}")
    else:
        print("   ✗ No model file found (need model.safetensors or pytorch_model.bin)")
        issues.append("Model weights file missing")
else:
    print(f"   ✗ Model directory not found: {model_path}")
    issues.append("Model directory missing")

# Check Wikipedia data
print("\n5. Checking Wikipedia data...")
wiki_path = os.getenv('WIKI_DATA_PATH', './data/wiki_texts.pkl')
if os.path.exists(wiki_path):
    size_mb = os.path.getsize(wiki_path) / (1024 * 1024)
    print(f"   ✓ Wikipedia data exists: {wiki_path} ({size_mb:.1f} MB)")
else:
    print(f"   ✗ Wikipedia data not found: {wiki_path}")
    issues.append("Wikipedia data missing")
    print("     Run: python prepare_data.py")

# Check port availability
print("\n6. Checking port availability...")
port = int(os.getenv('PORT', 5000))
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('localhost', port))
sock.close()

if result != 0:
    print(f"   ✓ Port {port} is available")
else:
    print(f"   ⚠ Port {port} is already in use")
    warnings.append(f"Port {port} in use")
    print(f"     Change PORT in .env or stop the process using port {port}")

# Check static files
print("\n7. Checking static files...")
static_files = ['static/index.html', 'static/style.css', 'static/app.js']
for file in static_files:
    if os.path.exists(file):
        print(f"   ✓ {file}")
    else:
        print(f"   ✗ {file} missing")
        issues.append(f"Missing static file: {file}")

# Summary
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

if not issues and not warnings:
    print("\n✓ All checks passed! You're ready to start the server.")
    print("\nRun: python start_simple.py")
elif not issues:
    print(f"\n⚠ {len(warnings)} warning(s) found:")
    for w in warnings:
        print(f"  - {w}")
    print("\nYou can still try to start the server:")
    print("  python start_simple.py")
else:
    print(f"\n✗ {len(issues)} issue(s) found:")
    for i in issues:
        print(f"  - {i}")
    
    if warnings:
        print(f"\n⚠ {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")
    
    print("\nPlease fix the issues above before starting the server.")
    print("\nFor help, see:")
    print("  - README.md")
    print("  - HOW_TO_START.md")
    print("  - TROUBLESHOOTING.md")
    
    sys.exit(1)

print()
