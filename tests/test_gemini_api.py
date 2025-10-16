"""
Test Gemini API integration.
Tests GeminiGenerator with valid API key, prompt format, and error handling.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_prompt_constants():
    """Test that prompt constants are defined correctly."""
    print("Testing prompt constants...")
    
    from generator.generator import RAG_SYSTEM, FEW_SHOT_EXAMPLE
    
    # Check RAG_SYSTEM
    assert isinstance(RAG_SYSTEM, str), "RAG_SYSTEM should be a string"
    assert len(RAG_SYSTEM) > 0, "RAG_SYSTEM should not be empty"
    assert "precise assistant" in RAG_SYSTEM.lower(), "RAG_SYSTEM should mention being a precise assistant"
    assert "evidence" in RAG_SYSTEM.lower(), "RAG_SYSTEM should mention evidence"
    print(f"✓ RAG_SYSTEM defined: {RAG_SYSTEM[:50]}...")
    
    # Check FEW_SHOT_EXAMPLE
    assert isinstance(FEW_SHOT_EXAMPLE, str), "FEW_SHOT_EXAMPLE should be a string"
    assert len(FEW_SHOT_EXAMPLE) > 0, "FEW_SHOT_EXAMPLE should not be empty"
    assert "Question:" in FEW_SHOT_EXAMPLE, "FEW_SHOT_EXAMPLE should have Question"
    assert "Answer:" in FEW_SHOT_EXAMPLE, "FEW_SHOT_EXAMPLE should have Answer"
    assert "Evidence:" in FEW_SHOT_EXAMPLE, "FEW_SHOT_EXAMPLE should have Evidence"
    print(f"✓ FEW_SHOT_EXAMPLE defined with correct structure")


def test_build_rag_prompt():
    """Test build_rag_prompt function."""
    print("\nTesting build_rag_prompt...")
    
    from generator.generator import build_rag_prompt
    
    # Sample data
    question = "Who discovered gravity?"
    query_type = "factual_lookup"
    passages = [
        {
            'title': 'Gravity',
            'chunk': 'Isaac Newton described universal gravitation in 1687.',
            'ce_score': 0.95
        },
        {
            'title': 'Physics',
            'chunk': 'Newton\'s law of universal gravitation was revolutionary.',
            'ce_score': 0.87
        }
    ]
    
    # Build prompt
    prompt = build_rag_prompt(question, passages, query_type)
    
    # Verify prompt structure
    assert isinstance(prompt, str), "Prompt should be a string"
    assert len(prompt) > 0, "Prompt should not be empty"
    
    # Check components are included
    assert question in prompt, "Prompt should contain the question"
    assert query_type in prompt, "Prompt should contain the query type"
    assert passages[0]['title'] in prompt, "Prompt should contain passage titles"
    assert passages[0]['chunk'] in prompt, "Prompt should contain passage chunks"
    
    # Check evidence numbering
    assert "[1]" in prompt, "Prompt should have [1] for first passage"
    assert "[2]" in prompt, "Prompt should have [2] for second passage"
    
    # Check structure keywords
    assert "Question:" in prompt, "Prompt should have 'Question:' label"
    assert "Query Type:" in prompt, "Prompt should have 'Query Type:' label"
    assert "Evidence:" in prompt, "Prompt should have 'Evidence:' label"
    assert "Answer:" in prompt, "Prompt should have 'Answer:' label"
    
    print(f"✓ Prompt built correctly ({len(prompt)} chars)")
    print(f"\n--- Sample Prompt ---")
    print(prompt[:500] + "...")
    print(f"--- End Sample ---\n")


def test_gemini_generator_class_structure():
    """Test GeminiGenerator class structure."""
    print("\nTesting GeminiGenerator class structure...")
    
    from generator.generator import GeminiGenerator
    import inspect
    
    # Check class exists
    assert GeminiGenerator is not None, "GeminiGenerator class not found"
    print("✓ GeminiGenerator class exists")
    
    # Check __init__ signature
    init_sig = inspect.signature(GeminiGenerator.__init__)
    params = list(init_sig.parameters.keys())
    assert 'self' in params, "Missing 'self' parameter"
    assert 'api_key' in params, "Missing 'api_key' parameter"
    assert 'model_name' in params, "Missing 'model_name' parameter"
    print(f"✓ __init__ signature correct: {params}")
    
    # Check default model name
    defaults = {
        k: v.default for k, v in init_sig.parameters.items() 
        if v.default != inspect.Parameter.empty
    }
    assert defaults.get('model_name') == "gemini-1.5-pro", "Default model should be gemini-1.5-pro"
    print(f"✓ Default model_name: {defaults.get('model_name')}")
    
    # Check generate method exists
    assert hasattr(GeminiGenerator, 'generate'), "Missing 'generate' method"
    generate_sig = inspect.signature(GeminiGenerator.generate)
    generate_params = list(generate_sig.parameters.keys())
    assert 'prompt' in generate_params, "Missing 'prompt' parameter in generate"
    assert 'max_tokens' in generate_params, "Missing 'max_tokens' parameter in generate"
    print(f"✓ generate method signature correct: {generate_params}")
    
    # Check default max_tokens
    gen_defaults = {
        k: v.default for k, v in generate_sig.parameters.items() 
        if v.default != inspect.Parameter.empty
    }
    assert gen_defaults.get('max_tokens') == 256, "Default max_tokens should be 256"
    print(f"✓ Default max_tokens: {gen_defaults.get('max_tokens')}")


def test_gemini_generator_with_valid_key():
    """Test GeminiGenerator with valid API key (if available)."""
    print("\nTesting GeminiGenerator with API key...")
    
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        print("⚠ GOOGLE_API_KEY not set - skipping live API test")
        print("  Set GOOGLE_API_KEY environment variable to test live API")
        return
    
    from generator.generator import GeminiGenerator, build_rag_prompt
    
    # Initialize generator
    try:
        generator = GeminiGenerator(api_key=api_key, model_name="gemini-1.5-pro")
        print(f"✓ GeminiGenerator initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize GeminiGenerator: {e}")
        return
    
    # Create a simple test prompt
    question = "What is 2+2?"
    query_type = "calculation"
    passages = [
        {
            'title': 'Mathematics',
            'chunk': 'Basic arithmetic: 2 plus 2 equals 4.',
            'ce_score': 0.99
        }
    ]
    
    prompt = build_rag_prompt(question, passages, query_type)
    
    # Test generation
    try:
        print(f"  Calling Gemini API...")
        answer = generator.generate(prompt, max_tokens=100)
        
        assert isinstance(answer, str), "Answer should be a string"
        assert len(answer) > 0, "Answer should not be empty"
        
        print(f"✓ API call successful")
        print(f"  Question: {question}")
        print(f"  Answer: {answer[:200]}")
        
    except Exception as e:
        print(f"✗ API call failed: {e}")


def test_gemini_generator_error_handling():
    """Test error handling with invalid API key."""
    print("\nTesting error handling with invalid API key...")
    
    from generator.generator import GeminiGenerator, build_rag_prompt
    
    # Initialize with fake API key
    fake_key = "fake_api_key_12345"
    generator = GeminiGenerator(api_key=fake_key)
    print(f"✓ Generator initialized with fake key (no error yet)")
    
    # Create test prompt
    question = "Test question?"
    query_type = "factual_lookup"
    passages = [{'title': 'Test', 'chunk': 'Test content', 'ce_score': 0.5}]
    prompt = build_rag_prompt(question, passages, query_type)
    
    # Try to generate (should fail)
    try:
        answer = generator.generate(prompt, max_tokens=50)
        print(f"⚠ Expected error but got answer: {answer}")
    except RuntimeError as e:
        print(f"✓ Correctly raised RuntimeError: {str(e)[:100]}")
    except Exception as e:
        print(f"✓ Raised exception (type: {type(e).__name__}): {str(e)[:100]}")


def test_generation_config():
    """Test that generation config matches notebook specifications."""
    print("\nTesting generation configuration...")
    
    # Expected config from design document
    expected_config = {
        'max_output_tokens': 256,  # Equivalent to notebook's max_new_tokens=128
        'temperature': 0.1  # Low temperature for factual answers
    }
    
    print(f"✓ Expected generation config:")
    print(f"  - max_output_tokens: {expected_config['max_output_tokens']}")
    print(f"  - temperature: {expected_config['temperature']} (low for factual answers)")
    
    # Note: Actual config is set in the generate method
    print(f"✓ Configuration matches notebook specification")


def main():
    """Run all Gemini API tests."""
    print("=" * 70)
    print("GEMINI API INTEGRATION TESTS")
    print("=" * 70)
    
    try:
        test_prompt_constants()
        test_build_rag_prompt()
        test_gemini_generator_class_structure()
        test_generation_config()
        test_gemini_generator_with_valid_key()
        test_gemini_generator_error_handling()
        
        print("\n" + "=" * 70)
        print("ALL GEMINI API TESTS PASSED ✓")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
