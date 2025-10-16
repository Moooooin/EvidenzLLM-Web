"""
Simple test script for model loading functionality.
Tests QueryClassifier loading from query_classifier_model directory.
"""

import os
import sys
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.models import QueryClassifier, load_query_classifier, label_map


def test_query_classifier_initialization():
    """Test QueryClassifier can be initialized."""
    print("Testing QueryClassifier initialization...")
    model = QueryClassifier(num_labels=4)
    assert model is not None
    assert hasattr(model, 'encoder')
    assert hasattr(model, 'dropout')
    assert hasattr(model, 'classifier')
    print("✓ QueryClassifier initialization successful")


def test_label_map_structure():
    """Test label_map has correct structure."""
    print("\nTesting label_map structure...")
    assert len(label_map) == 4
    assert "factual_lookup" in label_map
    assert "explanation" in label_map
    assert "reasoning" in label_map
    assert "calculation" in label_map
    assert label_map["factual_lookup"] == 0
    assert label_map["explanation"] == 1
    assert label_map["reasoning"] == 2
    assert label_map["calculation"] == 3
    print("✓ Label map structure correct")


def test_model_forward_pass():
    """Test model forward pass produces correct output shape."""
    print("\nTesting model forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    
    model = QueryClassifier(num_labels=4)
    model.to(device)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones(batch_size, seq_length).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Check output structure
    assert 'logits' in output
    assert 'loss' in output
    assert output['logits'].shape == (batch_size, 4)
    assert output['loss'] is None  # No labels provided
    print(f"✓ Model forward pass successful, output shape: {output['logits'].shape}")


def test_model_forward_with_labels():
    """Test model forward pass with labels computes loss."""
    print("\nTesting model forward pass with labels...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = QueryClassifier(num_labels=4)
    model.to(device)
    model.eval()
    
    # Create dummy input with labels
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones(batch_size, seq_length).to(device)
    labels = torch.tensor([0, 1]).to(device)
    
    # Forward pass
    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    # Check loss is computed
    assert output['loss'] is not None
    assert output['loss'].item() > 0
    print(f"✓ Model computed loss: {output['loss'].item():.4f}")


def test_load_query_classifier():
    """Test loading QueryClassifier from model directory."""
    print("\nTesting QueryClassifier loading from disk...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = './query_classifier_model'
    
    # Check if model directory exists
    if not os.path.exists(model_path):
        print(f"⚠ Model directory {model_path} not found - skipping")
        return
    
    # Check which format is available
    safetensors_path = os.path.join(model_path, 'model.safetensors')
    pytorch_bin_path = os.path.join(model_path, 'pytorch_model.bin')
    
    if os.path.exists(safetensors_path):
        print(f"  Found safetensors format at {safetensors_path}")
    elif os.path.exists(pytorch_bin_path):
        print(f"  Found pytorch_model.bin format at {pytorch_bin_path}")
    else:
        print(f"⚠ No model file found - skipping")
        return
    
    # Load model
    model = load_query_classifier(model_path, device)
    
    # Verify model is loaded correctly
    assert model is not None
    assert isinstance(model, QueryClassifier)
    assert next(model.parameters()).device.type == device.type
    
    # Verify model is in eval mode
    assert not model.training
    print("✓ Model loaded successfully and is in eval mode")


def test_loaded_model_inference():
    """Test that loaded model can perform inference."""
    print("\nTesting loaded model inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = './query_classifier_model'
    
    # Check if model directory exists
    if not os.path.exists(model_path):
        print(f"⚠ Model directory {model_path} not found - skipping")
        return
    
    # Load model
    try:
        model = load_query_classifier(model_path, device)
    except FileNotFoundError:
        print(f"⚠ Model files not found - skipping")
        return
    
    # Create sample input
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
    
    question = "What is machine learning?"
    inputs = tokenizer(question, return_tensors='pt').to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
    
    # Verify output
    assert 'logits' in output
    assert output['logits'].shape == (1, 4)
    
    # Get prediction
    pred = torch.argmax(output['logits'], dim=-1).item()
    assert pred in [0, 1, 2, 3]
    
    # Map to label
    reverse_label_map = {v: k for k, v in label_map.items()}
    predicted_label = reverse_label_map[pred]
    
    print(f"✓ Model inference successful")
    print(f"  Question: '{question}'")
    print(f"  Predicted class: {pred} ({predicted_label})")
    print(f"  Logits: {output['logits'][0].tolist()}")


def test_model_loading_error_handling():
    """Test error handling when model files are missing."""
    print("\nTesting error handling for missing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fake_path = './nonexistent_model_directory'
    
    try:
        load_query_classifier(fake_path, device)
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        print(f"✓ Correctly raised FileNotFoundError: {e}")


def main():
    """Run all tests."""
    print("=" * 70)
    print("MODEL LOADING TESTS")
    print("=" * 70)
    
    try:
        test_query_classifier_initialization()
        test_label_map_structure()
        test_model_forward_pass()
        test_model_forward_with_labels()
        test_load_query_classifier()
        test_loaded_model_inference()
        test_model_loading_error_handling()
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
