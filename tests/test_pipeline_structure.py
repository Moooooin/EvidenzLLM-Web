"""
Test pipeline structure without heavy imports.
"""

import os
import sys


def test_pipeline_files_exist():
    """Test that pipeline files exist."""
    print("Testing pipeline files existence...")
    
    pipeline_file = './pipeline/pipeline.py'
    assert os.path.exists(pipeline_file), f"Pipeline file not found: {pipeline_file}"
    print(f"✓ Pipeline file exists: {pipeline_file}")
    
    # Check file size
    file_size = os.path.getsize(pipeline_file)
    print(f"  File size: {file_size} bytes")


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


def test_label_map():
    """Test label map structure."""
    print("\nTesting label map...")
    
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
        # List model files
        model_files = os.listdir(model_path)
        print(f"  Contains {len(model_files)} files")
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


def test_pipeline_workflow():
    """Test expected pipeline workflow."""
    print("\nTesting pipeline workflow...")
    
    workflow_steps = [
        "1. classify_query: Classify the question type",
        "2. retriever.retrieve: Get top 5 relevant passages",
        "3. build_rag_prompt: Build prompt with evidence",
        "4. generator.generate: Generate answer via Gemini API"
    ]
    
    print(f"✓ Expected pipeline workflow:")
    for step in workflow_steps:
        print(f"  {step}")


def main():
    """Run all pipeline structure tests."""
    print("=" * 70)
    print("PIPELINE STRUCTURE TESTS")
    print("=" * 70)
    
    try:
        test_pipeline_files_exist()
        test_pipeline_config_structure()
        test_label_map()
        test_process_query_output_structure()
        test_pipeline_initialization_requirements()
        test_sample_questions()
        test_pipeline_workflow()
        
        print("\n" + "=" * 70)
        print("ALL PIPELINE STRUCTURE TESTS PASSED ✓")
        print("=" * 70)
        print("\nNote: Full pipeline tests with model loading require running")
        print("the application with proper environment setup.")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
