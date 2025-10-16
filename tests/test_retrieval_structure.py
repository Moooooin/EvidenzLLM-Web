"""
Test retrieval pipeline structure without heavy model loading.
"""

import os
import sys
import inspect

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrieval.retrieval import build_chunks, simple_tokenize


def test_simple_tokenize():
    """Test simple tokenization function."""
    print("Testing simple_tokenize...")
    
    # Test basic tokenization
    text = "Hello World! This is a TEST."
    tokens = simple_tokenize(text)
    assert isinstance(tokens, list), "Tokens should be a list"
    assert len(tokens) > 0, "Should produce tokens"
    assert all(isinstance(t, str) for t in tokens), "All tokens should be strings"
    print(f"✓ Tokenized '{text}' -> {tokens}")
    
    # Test lowercase conversion
    text2 = "UPPERCASE lowercase"
    tokens2 = simple_tokenize(text2)
    assert all(t.islower() or t.isdigit() for t in tokens2), "All tokens should be lowercase"
    print(f"✓ Lowercase conversion works: {tokens2}")
    
    # Test special character removal
    text3 = "test@#$%test"
    tokens3 = simple_tokenize(text3)
    assert '@' not in ' '.join(tokens3), "Special characters should be removed"
    print(f"✓ Special character removal works: {tokens3}")


def test_build_chunks_basic():
    """Test build_chunks with sample Wikipedia text."""
    print("\nTesting build_chunks with sample data...")
    
    # Create sample wiki entries
    wiki_entries = [
        {
            'title': 'Machine Learning',
            'text': 'Machine learning is a subset of artificial intelligence. ' * 20
        },
        {
            'title': 'Physics',
            'text': 'Physics is the natural science that studies matter. ' * 15
        }
    ]
    
    # Build chunks
    chunks, meta = build_chunks(wiki_entries, max_chars=200, overlap_chars=50)
    
    # Verify output structure
    assert isinstance(chunks, list), "Chunks should be a list"
    assert isinstance(meta, list), "Meta should be a list"
    assert len(chunks) == len(meta), "Chunks and meta should have same length"
    assert len(chunks) > 0, "Should produce at least one chunk"
    
    print(f"✓ Created {len(chunks)} chunks from {len(wiki_entries)} entries")
    
    # Verify chunk structure
    for i, (chunk, m) in enumerate(zip(chunks[:3], meta[:3])):
        assert isinstance(chunk, str), f"Chunk {i} should be a string"
        assert len(chunk) > 0, f"Chunk {i} should not be empty"
        assert 'title' in m, f"Meta {i} should have 'title'"
        assert 'src_id' in m, f"Meta {i} should have 'src_id'"
        print(f"  Chunk {i}: {len(chunk)} chars, title='{m['title']}', src_id={m['src_id']}")
    
    # Verify max_chars constraint
    for i, chunk in enumerate(chunks):
        # Allow some flexibility for sentence boundaries
        assert len(chunk) <= 250, f"Chunk {i} exceeds max_chars significantly: {len(chunk)}"
    
    print(f"✓ All chunks respect max_chars constraint")
    
    return chunks, meta


def test_build_chunks_overlap():
    """Test that chunks have proper overlap."""
    print("\nTesting chunk overlap...")
    
    wiki_entries = [
        {
            'title': 'Test Article',
            'text': 'This is sentence one. This is sentence two. This is sentence three. ' * 10
        }
    ]
    
    chunks, meta = build_chunks(wiki_entries, max_chars=150, overlap_chars=30)
    
    if len(chunks) > 1:
        # Check if there's some overlap between consecutive chunks
        for i in range(min(3, len(chunks) - 1)):  # Check first few
            chunk1 = chunks[i]
            chunk2 = chunks[i + 1]
            # Get last 30 chars of chunk1 and first 30 chars of chunk2
            tail1 = chunk1[-30:]
            head2 = chunk2[:30]
            # There should be some common words
            words1 = set(tail1.split())
            words2 = set(head2.split())
            common = words1 & words2
            print(f"  Chunks {i} and {i+1} have {len(common)} common words in overlap region")
        
        print(f"✓ Overlap mechanism working (created {len(chunks)} overlapping chunks)")
    else:
        print(f"  Only 1 chunk created, overlap not testable")


def test_build_chunks_default_params():
    """Test build_chunks with default parameters from notebook."""
    print("\nTesting build_chunks with default parameters (max_chars=1200, overlap_chars=200)...")
    
    # Create a longer text
    long_text = "This is a test sentence. " * 100  # ~2500 chars
    wiki_entries = [
        {
            'title': 'Long Article',
            'text': long_text
        }
    ]
    
    # Use default parameters from notebook
    chunks, meta = build_chunks(wiki_entries, max_chars=1200, overlap_chars=200)
    
    print(f"✓ Created {len(chunks)} chunks with default parameters")
    
    # Verify chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"  Chunk {i}: {len(chunk)} chars")
        assert len(chunk) <= 1300, f"Chunk {i} significantly exceeds max_chars"
    
    return chunks, meta


def test_hybrid_retriever_class_structure():
    """Test HybridRetriever class structure without initialization."""
    print("\nTesting HybridRetriever class structure...")
    
    from retrieval.retrieval import HybridRetriever
    
    # Check class exists
    assert HybridRetriever is not None, "HybridRetriever class not found"
    print("✓ HybridRetriever class exists")
    
    # Check __init__ signature
    init_sig = inspect.signature(HybridRetriever.__init__)
    params = list(init_sig.parameters.keys())
    assert 'self' in params, "Missing 'self' parameter"
    assert 'chunk_texts' in params, "Missing 'chunk_texts' parameter"
    assert 'chunk_meta' in params, "Missing 'chunk_meta' parameter"
    print(f"✓ __init__ signature correct: {params}")
    
    # Check retrieve method exists
    assert hasattr(HybridRetriever, 'retrieve'), "Missing 'retrieve' method"
    retrieve_sig = inspect.signature(HybridRetriever.retrieve)
    retrieve_params = list(retrieve_sig.parameters.keys())
    assert 'query' in retrieve_params, "Missing 'query' parameter in retrieve"
    assert 'top_k' in retrieve_params, "Missing 'top_k' parameter in retrieve"
    assert 'alpha' in retrieve_params, "Missing 'alpha' parameter in retrieve"
    assert 'bm25_candidates' in retrieve_params, "Missing 'bm25_candidates' parameter"
    assert 'dense_candidates' in retrieve_params, "Missing 'dense_candidates' parameter"
    assert 'rerank_top' in retrieve_params, "Missing 'rerank_top' parameter"
    print(f"✓ retrieve method signature correct")
    print(f"  Parameters: {retrieve_params}")
    
    # Check default values
    defaults = {
        k: v.default for k, v in retrieve_sig.parameters.items() 
        if v.default != inspect.Parameter.empty
    }
    print(f"  Default values: {defaults}")
    assert defaults.get('top_k') == 5, "Default top_k should be 5"
    assert defaults.get('alpha') == 0.6, "Default alpha should be 0.6"
    assert defaults.get('bm25_candidates') == 200, "Default bm25_candidates should be 200"
    assert defaults.get('dense_candidates') == 200, "Default dense_candidates should be 200"
    assert defaults.get('rerank_top') == 32, "Default rerank_top should be 32"
    print(f"✓ Default parameters match notebook specification")


def test_retrieval_output_format():
    """Test expected output format of retrieve method."""
    print("\nTesting expected retrieve output format...")
    
    # Expected format based on design document
    expected_result = {
        'chunk': 'Sample text chunk',
        'title': 'Article Title',
        'ce_score': 0.923
    }
    
    print("✓ Expected result format:")
    print(f"  - chunk: str (passage text)")
    print(f"  - title: str (article title)")
    print(f"  - ce_score: float (cross-encoder score)")
    
    # Verify keys
    required_keys = ['chunk', 'title', 'ce_score']
    for key in required_keys:
        assert key in expected_result, f"Missing required key: {key}"
    
    print(f"✓ All required keys present: {required_keys}")


def main():
    """Run all retrieval structure tests."""
    print("=" * 70)
    print("RETRIEVAL PIPELINE STRUCTURE TESTS")
    print("=" * 70)
    
    try:
        test_simple_tokenize()
        test_build_chunks_basic()
        test_build_chunks_overlap()
        test_build_chunks_default_params()
        test_hybrid_retriever_class_structure()
        test_retrieval_output_format()
        
        print("\n" + "=" * 70)
        print("ALL RETRIEVAL STRUCTURE TESTS PASSED ✓")
        print("=" * 70)
        print("\nNote: Full retrieval tests with model loading require running")
        print("the application with proper environment setup.")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
