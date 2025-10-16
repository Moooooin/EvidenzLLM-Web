#!/usr/bin/env python
"""
Test the prompt generation to see what's being sent to Gemini.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, '.')

from generator.generator import build_rag_prompt

# Sample data
question = "What is machine learning?"
query_type = "explanation"
passages = [
    {
        'title': 'Machine Learning',
        'chunk': 'Machine learning is a subset of artificial intelligence that focuses on building systems that can learn from data.',
        'ce_score': 0.95
    },
    {
        'title': 'AI Overview',
        'chunk': 'Artificial intelligence encompasses various techniques including machine learning, which allows computers to improve through experience.',
        'ce_score': 0.87
    }
]

print("Testing prompt generation...")
print("=" * 70)
print()

try:
    prompt = build_rag_prompt(question, passages, query_type)
    
    print("✓ Prompt generated successfully")
    print(f"✓ Prompt length: {len(prompt)} characters")
    print()
    print("=" * 70)
    print("GENERATED PROMPT:")
    print("=" * 70)
    print(prompt)
    print("=" * 70)
    print()
    
    # Check for potential issues
    issues = []
    
    if len(prompt) == 0:
        issues.append("Prompt is empty")
    
    if len(prompt) > 30000:
        issues.append(f"Prompt is very long ({len(prompt)} chars)")
    
    # Check for non-printable characters
    non_printable = [c for c in prompt if ord(c) < 32 and c not in '\n\r\t']
    if non_printable:
        issues.append(f"Contains {len(non_printable)} non-printable characters")
    
    if issues:
        print("⚠ Potential issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ No obvious issues with prompt format")
    
    print()
    print("Now testing with Gemini API...")
    print()
    
    # Test with actual API
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("✗ GOOGLE_API_KEY not set")
        sys.exit(1)
    
    from generator.generator import GeminiGenerator
    
    generator = GeminiGenerator(api_key=api_key, model_name='gemini-2.0-flash-exp')
    
    print("Sending to Gemini...")
    response = generator.generate(prompt, max_tokens=100)
    
    print("✓ API call successful!")
    print()
    print("Response:")
    print("-" * 70)
    print(response)
    print("-" * 70)
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
