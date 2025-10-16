#!/usr/bin/env python
"""
Test if your Google API key works with Gemini.
"""

import os
import sys

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

api_key = os.getenv('GOOGLE_API_KEY')

if not api_key:
    print("✗ GOOGLE_API_KEY not set")
    print("Run: source load_env.sh")
    sys.exit(1)

print(f"Testing API key: {api_key[:10]}...")
print()

try:
    import google.generativeai as genai
    
    # Configure API
    genai.configure(api_key=api_key)
    
    # Try to create a model
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    # Try a simple generation
    print("Sending test request to Gemini...")
    response = model.generate_content("Say 'Hello, I am working!'")
    
    print("✓ API key is valid!")
    print(f"✓ Response: {response.text}")
    print()
    print("Your API key works correctly.")
    
except Exception as e:
    print(f"✗ API key test failed!")
    print(f"Error: {str(e)}")
    print()
    print("Possible issues:")
    print("1. API key is invalid or revoked")
    print("2. Gemini API not enabled in Google Cloud Console")
    print("3. API key doesn't have permission for Gemini API")
    print()
    print("To fix:")
    print("1. Go to https://makersuite.google.com/app/apikey")
    print("2. Create a new API key")
    print("3. Update your .env file with the new key")
    print("4. Run: source load_env.sh")
    print("5. Restart the server")
    sys.exit(1)
