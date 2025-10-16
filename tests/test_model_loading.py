"""
Test model loading functionality.
Tests QueryClassifier loading from query_classifier_model directory.
"""

import os
import sys
import torch
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.models import QueryClassifier, load_query_classifier, label_map


class TestModelLoading:
    """Test suite for model loading functionality."""
    
    @pytest.fixture
    def device(self):
        """Get device for testing."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def model_path(self):
        """Get path to model directory."""
        return './query_classifier_model'
    
    def test_query_classifier_initialization(self):
        """Test QueryClassifier can be initialized."""
        model = QueryClassifier(num_labels=4)
        assert model is not None
        assert hasattr(model, 'encoder')
        assert hasattr(model, 'dropout')
        assert hasattr(model, 'classifier')
    
    def test_label_map_structure(self):
        """Test label_map has correct structure."""
        assert len(label_map) == 4
        assert "factual_lookup" in label_map
        assert "explanation" in label_map
        assert "reasoning" in label_map
        assert "calculation" in label_map
        assert label_map["factual_lookup"] == 0
        assert label_map["explanation"] == 1
        assert label_map["reasoning"] == 2
        assert label_map["calculation"] == 3
    
    def test_model_forward_pass(self, device):
        """Test model forward pass produces correct output shape."""
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
    
    def test_model_forward_with_labels(self, device):
        """Test model forward pass with labels computes loss."""
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
    
    def test_load_query_classifier_from_safetensors(self, device, model_path):
        """Test loading QueryClassifier from safetensors format."""
        # Check if model directory exists
        if not os.path.exists(model_path):
            pytest.skip(f"Model directory {model_path} not found")
        
        safetensors_path = os.path.join(model_path, 'model.safetensors')
        if not os.path.exists(safetensors_path):
            pytest.skip(f"Safetensors file not found at {safetensors_path}")
        
        # Load model
        model = load_query_classifier(model_path, device)
        
        # Verify model is loaded correctly
        assert model is not None
        assert isinstance(model, QueryClassifier)
        assert next(model.parameters()).device.type == device.type
        
        # Verify model is in eval mode
        assert not model.training
    
    def test_load_query_classifier_from_pytorch_bin(self, device, model_path):
        """Test loading QueryClassifier from pytorch_model.bin format."""
        # Check if model directory exists
        if not os.path.exists(model_path):
            pytest.skip(f"Model directory {model_path} not found")
        
        pytorch_bin_path = os.path.join(model_path, 'pytorch_model.bin')
        if not os.path.exists(pytorch_bin_path):
            pytest.skip(f"PyTorch bin file not found at {pytorch_bin_path}")
        
        # Load model
        model = load_query_classifier(model_path, device)
        
        # Verify model is loaded correctly
        assert model is not None
        assert isinstance(model, QueryClassifier)
        assert next(model.parameters()).device.type == device.type
        
        # Verify model is in eval mode
        assert not model.training
    
    def test_loaded_model_inference(self, device, model_path):
        """Test that loaded model can perform inference."""
        # Check if model directory exists
        if not os.path.exists(model_path):
            pytest.skip(f"Model directory {model_path} not found")
        
        # Load model
        model = load_query_classifier(model_path, device)
        
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
    
    def test_model_loading_error_handling(self, device):
        """Test error handling when model files are missing."""
        fake_path = './nonexistent_model_directory'
        
        with pytest.raises(FileNotFoundError):
            load_query_classifier(fake_path, device)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
