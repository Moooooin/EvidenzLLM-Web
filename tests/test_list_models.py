#!/usr/bin/env python3
"""
Quick test script to list all available Gemini models.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure API
api_key = os.getenv('GOOGLE_API_KEY')
if not api_key:
    print("ERROR: GOOGLE_API_KEY not found in .env file")
    exit(1)

genai.configure(api_key=api_key)

print("=" * 80)
print(f"Google Generative AI SDK Version: {genai.__version__}")
print("=" * 80)
print("\nListing all available models...\n")

# List all models
models = genai.list_models()

print(f"{'Model Name':<40} {'Supported Methods':<30} {'API Version'}")
print("-" * 80)

for model in models:
    # Get model name
    name = model.name.replace('models/', '')
    
    # Get supported generation methods
    methods = []
    if hasattr(model, 'supported_generation_methods'):
        methods = model.supported_generation_methods
    
    # Determine API version (v1 for stable, v1beta for experimental)
    api_version = "v1 (stable)" if not any(x in name for x in ['-exp', '-latest', 'experimental']) else "v1beta"
    
    # Format methods
    methods_str = ', '.join(methods) if methods else 'N/A'
    if len(methods_str) > 28:
        methods_str = methods_str[:25] + '...'
    
    print(f"{name:<40} {methods_str:<30} {api_version}")

print("\n" + "=" * 80)
print("\nModels that support 'generateContent' (text generation):\n")

# Re-fetch models for this section
models_list = list(genai.list_models())
text_gen_models = []

for model in models_list:
    if hasattr(model, 'supported_generation_methods'):
        if 'generateContent' in model.supported_generation_methods:
            name = model.name.replace('models/', '')
            text_gen_models.append(name)
            api_version = "v1" if not any(x in name for x in ['-exp', '-latest', 'experimental']) else "v1beta"
            print(f"  ✓ {name:<50} [{api_version}]")

print(f"\nTotal: {len(text_gen_models)} models")

print("\n" + "=" * 80)
print("\nRecommended models for your use case:")
print("  • gemini-1.5-flash (fast, stable, good BLOCK_NONE support)")
print("  • gemini-1.5-pro (more capable, stable)")
print("  • gemini-2.0-flash-exp (experimental, latest features)")
print("=" * 80)
