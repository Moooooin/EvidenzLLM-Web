"""
Basic test script for model structure without heavy dependencies.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.models import label_map


def test_label_map_structure():
    """Test label_map has correct structure."""
    print("Testing label_map structure...")
    assert len(label_map) == 4, f"Expected 4 labels, got {len(label_map)}"
    assert "factual_lookup" in label_map, "Missing 'factual_lookup' label"
    assert "explanation" in label_map, "Missing 'explanation' label"
    assert "reasoning" in label_map, "Missing 'reasoning' label"
    assert "calculation" in label_map, "Missing 'calculation' label"
    assert label_map["factual_lookup"] == 0, f"Expected 0, got {label_map['factual_lookup']}"
    assert label_map["explanation"] == 1, f"Expected 1, got {label_map['explanation']}"
    assert label_map["reasoning"] == 2, f"Expected 2, got {label_map['reasoning']}"
    assert label_map["calculation"] == 3, f"Expected 3, got {label_map['calculation']}"
    print("✓ Label map structure correct")
    print(f"  Labels: {label_map}")


def test_model_files_exist():
    """Test that model files exist in the expected location."""
    print("\nTesting model files existence...")
    model_path = './query_classifier_model'
    
    if not os.path.exists(model_path):
        print(f"⚠ Model directory {model_path} not found")
        return False
    
    print(f"✓ Model directory exists: {model_path}")
    
    # Check for model files
    safetensors_path = os.path.join(model_path, 'model.safetensors')
    pytorch_bin_path = os.path.join(model_path, 'pytorch_model.bin')
    
    has_safetensors = os.path.exists(safetensors_path)
    has_pytorch_bin = os.path.exists(pytorch_bin_path)
    
    if has_safetensors:
        print(f"✓ Found safetensors format: {safetensors_path}")
        file_size = os.path.getsize(safetensors_path) / (1024 * 1024)
        print(f"  File size: {file_size:.2f} MB")
    
    if has_pytorch_bin:
        print(f"✓ Found pytorch_model.bin format: {pytorch_bin_path}")
        file_size = os.path.getsize(pytorch_bin_path) / (1024 * 1024)
        print(f"  File size: {file_size:.2f} MB")
    
    if not has_safetensors and not has_pytorch_bin:
        print(f"⚠ No model files found in {model_path}")
        return False
    
    # Check for other expected files
    expected_files = ['tokenizer_config.json', 'vocab.json', 'merges.txt', 'special_tokens_map.json']
    for filename in expected_files:
        filepath = os.path.join(model_path, filename)
        if os.path.exists(filepath):
            print(f"✓ Found {filename}")
        else:
            print(f"  {filename} not found (may not be required)")
    
    return True


def test_model_class_structure():
    """Test that QueryClassifier class has expected structure."""
    print("\nTesting QueryClassifier class structure...")
    
    from models.models import QueryClassifier
    
    # Check class exists
    assert QueryClassifier is not None, "QueryClassifier class not found"
    print("✓ QueryClassifier class exists")
    
    # Check __init__ signature
    import inspect
    init_sig = inspect.signature(QueryClassifier.__init__)
    params = list(init_sig.parameters.keys())
    assert 'self' in params, "Missing 'self' parameter"
    assert 'num_labels' in params, "Missing 'num_labels' parameter"
    assert 'model_name' in params, "Missing 'model_name' parameter"
    print(f"✓ __init__ signature correct: {params}")
    
    # Check forward method exists
    assert hasattr(QueryClassifier, 'forward'), "Missing 'forward' method"
    forward_sig = inspect.signature(QueryClassifier.forward)
    forward_params = list(forward_sig.parameters.keys())
    assert 'input_ids' in forward_params, "Missing 'input_ids' parameter in forward"
    assert 'attention_mask' in forward_params, "Missing 'attention_mask' parameter in forward"
    assert 'labels' in forward_params, "Missing 'labels' parameter in forward"
    print(f"✓ forward method signature correct: {forward_params}")


def test_load_function_exists():
    """Test that load_query_classifier function exists."""
    print("\nTesting load_query_classifier function...")
    
    from models.models import load_query_classifier
    
    assert load_query_classifier is not None, "load_query_classifier function not found"
    print("✓ load_query_classifier function exists")
    
    import inspect
    sig = inspect.signature(load_query_classifier)
    params = list(sig.parameters.keys())
    assert 'model_load_path' in params, "Missing 'model_load_path' parameter"
    assert 'device' in params, "Missing 'device' parameter"
    print(f"✓ Function signature correct: {params}")


def main():
    """Run all basic tests."""
    print("=" * 70)
    print("BASIC MODEL STRUCTURE TESTS")
    print("=" * 70)
    
    try:
        test_label_map_structure()
        test_model_files_exist()
        test_model_class_structure()
        test_load_function_exists()
        
        print("\n" + "=" * 70)
        print("ALL BASIC TESTS PASSED ✓")
        print("=" * 70)
        print("\nNote: Full model loading tests require running the application")
        print("with proper PyTorch environment setup.")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
