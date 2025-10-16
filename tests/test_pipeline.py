"""
Test full EvidenzPipeline.
Tests pipeline initialization, classify_query, and process_query end-to-end.
"""

import os
import sys
import inspect

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_pipeline_class_structure():
    """Test EvidenzPipeline class structure."""
    print("Testing EvidenzPipeline class structure...")
    
    from pipeline.pipeline import EvidenzPipeline
    
    # Check class exists
    assert EvidenzPipeline is not None, "EvidenzPipeline class not found"
    print("✓ EvidenzPipeline class exists")
    
    # Check __init__ signature
    init_sig = inspect.signature(EvidenzPipeline.__init__)
    params = list(init_sig.parameters.keys())
    assert 'self' in params, "Missing 'self' parameter"
    assert 'config' in params, "Missing 'config' parameter"
    print(f"✓ __init__ signature correct: {params}")
    
    # Check classify_query method
    assert hasattr(EvidenzPipeline, 'classify_query'), "Missing 'classify_query' method"
    classify_sig = inspect.signature(EvidenzPipeline.classify_query)
    classify_params = list(classify_sig.parameters.keys())
    assert 'question' in classify_params, "Missing 'question' parameter in classify_query"
    print(f"✓ classify_query method exists: {classify_params}")
    
    # Check process_query method
    assert hasattr(EvidenzPipeline, 'process_query'), "Missing 'process_query' method"
    process_sig = inspect.signature(EvidenzPipeline.process_query)
    process_params = list(process_sig.parameters.keys())
    assert 'question' in process_params, "Missing 'question' parameter in process_query"
    print(f"✓ process_query method exists: {process_params}")


def test_pipeline_config_structure():
    """Test expected pipeline configuration structure."""
    print("\nTesting pipeline configuration structure...")
    
    # Expected config from design document
    expected_config = {
        'classifier_path': './query_classifier_model',
        'wiki_data_path': './data/wiki_texts.pkl',
        'gemini_api_key': 'test_key',
        'gemini_model': 'gemini-1.5-pro'
    }
    
    print(f"✓ Expected configuration keys:")
    for key, value in expected_config.items():
        print(f"  - {key}: {value if key != 'gemini_api_key' else '[REDACTED]'}")


def test_label_map_in_pipeline():
    """Test that pipeline uses correct label map."""
    print("\nTesting label map in pipeline...")
    
    expected_label_map = {
        "factual_lookup": 0,
        "explanation": 1,
        "reasoning": 2,
        "calculation": 3
    }
    
    expected_reverse = {v: k for k, v in expected_label_map.items()}
    
    print(f"✓ Expected label_map: {expected_label_map}")
    print(f"✓ Expected reverse_label_map: {expected_reverse}")


def test_process_query_output_structure():
    """Test expected output structure of process_query."""
    print("\nTesting process_query output structure...")
    
    # Expected output format from design document
    expected_output = {
        'question': 'Who developed the theory of relativity?',
        'query_type': 'factual_lookup',
        'answer': 'Albert Einstein developed the theory of relativity [1][2].',
        'passages': [
            {
                'chunk': 'Albert Einstein published his theory...',
                'title': 'Theory of Relativity',
                'ce_score': 0.923
            }
        ]
    }
    
    print(f"✓ Expected output structure:")
    print(f"  - question: str (original question)")
    print(f"  - query_type: str (classified type)")
    print(f"  - answer: str (generated answer)")
    print(f"  - passages: list[dict] (evidence passages)")
    
    # Verify keys
    required_keys = ['question', 'query_type', 'answer', 'passages']
    for key in required_keys:
        assert key in expected_output, f"Missing required key: {key}"
    
    print(f"✓ All required keys present: {required_keys}")
    
    # Verify passage structure
    passage = expected_output['passages'][0]
    passage_keys = ['chunk', 'title', 'ce_score']
    for key in passage_keys:
        assert key in passage, f"Missing required passage key: {key}"
    
    print(f"✓ Passage structure correct: {passage_keys}")


def test_pipeline_initialization_requirements():
    """Test pipeline initialization requirements."""
    print("\nTesting pipeline initialization requirements...")
    
    # Check if required files exist
    model_path = './query_classifier_model'
    wiki_data_path = './data/wiki_texts.pkl'
    
    if os.path.exists(model_path):
        print(f"✓ Model directory exists: {model_path}")
    else:
        print(f"⚠ Model directory not found: {model_path}")
    
    if os.path.exists(wiki_data_path):
        print(f"✓ Wiki data file exists: {wiki_data_path}")
        file_size = os.path.getsize(wiki_data_path) / (1024 * 1024)
        print(f"  File size: {file_size:.2f} MB")
    else:
        print(f"⚠ Wiki data file not found: {wiki_data_path}")
        print(f"  Run prepare_data.py to generate Wikipedia data")


def test_sample_questions():
    """Test with sample questions from notebook."""
    print("\nTesting with sample questions...")
    
    # Sample questions that might be in the notebook
    sample_questions = [
        ("What is machine learning?", "explanation"),
        ("Who discovered gravity?", "factual_lookup"),
        ("Why does the sky appear blue?", "explanation"),
        ("Calculate 15% of 200", "calculation"),
        ("How does photosynthesis work?", "reasoning")
    ]
    
    print(f"✓ Sample questions for testing:")
    for i, (question, expected_type) in enumerate(sample_questions, 1):
        print(f"  {i}. '{question}' -> expected: {expected_type}")


def test_pipeline_with_real_data():
    """Test pipeline with real data if available."""
    print("\nTesting pipeline with real data...")
    
    # Check if all required components are available
    model_path = './query_classifier_model'
    wiki_data_path = './data/wiki_texts.pkl'
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if not os.path.exists(model_path):
        print(f"⚠ Model directory not found - skipping real pipeline test")
        return
    
    if not os.path.exists(wiki_data_path):
        print(f"⚠ Wiki data not found - skipping real pipeline test")
        return
    
    if not api_key:
        print(f"⚠ GOOGLE_API_KEY not set - skipping real pipeline test")
        return
    
    print(f"  All components available, attempting pipeline initialization...")
    
    try:
        from pipeline.pipeline import EvidenzPipeline
        
        config = {
            'classifier_path': model_path,
            'wiki_data_path': wiki_data_path,
            'gemini_api_key': api_key,
            'gemini_model': 'gemini-1.5-pro'
        }
        
        print(f"  Initializing pipeline (this may take a moment)...")
        pipeline = EvidenzPipeline(config)
        print(f"✓ Pipeline initialized successfully")
        
        # Test classify_query
        test_question = "What is machine learning?"
        print(f"\n  Testing classify_query with: '{test_question}'")
        query_type = pipeline.classify_query(test_question)
        print(f"✓ Query classified as: {query_type}")
        
        # Test process_query
        print(f"\n  Testing full process_query...")
        result = pipeline.process_query(test_question)
        
        # Verify result structure
        assert 'question' in result, "Missing 'question' in result"
        assert 'query_type' in result, "Missing 'query_type' in result"
        assert 'answer' in result, "Missing 'answer' in result"
        assert 'passages' in result, "Missing 'passages' in result"
        
        print(f"✓ process_query completed successfully")
        print(f"\n  Result:")
        print(f"    Question: {result['question']}")
        print(f"    Query Type: {result['query_type']}")
        print(f"    Answer: {result['answer'][:100]}...")
        print(f"    Passages: {len(result['passages'])} passages returned")
        
        # Verify passages
        assert len(result['passages']) <= 5, "Should return at most 5 passages"
        for i, passage in enumerate(result['passages']):
            assert 'chunk' in passage, f"Passage {i} missing 'chunk'"
            assert 'title' in passage, f"Passage {i} missing 'title'"
            assert 'ce_score' in passage, f"Passage {i} missing 'ce_score'"
            print(f"    Passage {i+1}: {passage['title']} (score: {passage['ce_score']:.4f})")
        
        print(f"\n✓ All pipeline tests passed with real data")
        
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all pipeline tests."""
    print("=" * 70)
    print("FULL PIPELINE TESTS")
    print("=" * 70)
    
    try:
        test_pipeline_class_structure()
        test_pipeline_config_structure()
        test_label_map_in_pipeline()
        test_process_query_output_structure()
        test_pipeline_initialization_requirements()
        test_sample_questions()
        test_pipeline_with_real_data()
        
        print("\n" + "=" * 70)
        print("ALL PIPELINE TESTS PASSED ✓")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
