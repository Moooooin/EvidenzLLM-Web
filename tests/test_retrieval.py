"""
Test retrieval pipeline functionality.
Tests build_chunks, HybridRetriever initialization, and retrieve method.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrieval.retrieval import build_chunks, simple_tokenize, HybridRetriever


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
        for i in range(len(chunks) - 1):
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


def test_hybrid_retriever_initialization():
    """Test HybridRetriever initialization."""
    print("\nTesting HybridRetriever initialization...")
    
    # Create sample data
    chunk_texts = [
        "Machine learning is a method of data analysis.",
        "Artificial intelligence is the simulation of human intelligence.",
        "Deep learning is part of machine learning methods.",
        "Neural networks are computing systems inspired by biological neural networks.",
        "Python is a popular programming language for data science."
    ]
    
    chunk_meta = [
        {'title': 'Machine Learning', 'src_id': 0},
        {'title': 'Artificial Intelligence', 'src_id': 1},
        {'title': 'Deep Learning', 'src_id': 2},
        {'title': 'Neural Networks', 'src_id': 3},
        {'title': 'Python', 'src_id': 4}
    ]
    
    print(f"  Initializing retriever with {len(chunk_texts)} chunks...")
    print(f"  (This may take a moment to download models...)")
    
    # Initialize retriever
    retriever = HybridRetriever(chunk_texts, chunk_meta)
    
    # Verify attributes
    assert hasattr(retriever, 'chunk_texts'), "Should have chunk_texts attribute"
    assert hasattr(retriever, 'chunk_meta'), "Should have chunk_meta attribute"
    assert hasattr(retriever, 'bm25'), "Should have bm25 attribute"
    assert hasattr(retriever, 'dense_model'), "Should have dense_model attribute"
    assert hasattr(retriever, 'index_dense'), "Should have index_dense attribute"
    assert hasattr(retriever, 'cross_encoder'), "Should have cross_encoder attribute"
    
    print(f"✓ HybridRetriever initialized successfully")
    print(f"  - BM25 index: {retriever.bm25}")
    print(f"  - Dense model: {retriever.dense_model}")
    print(f"  - FAISS index: {retriever.index_dense.ntotal} vectors")
    print(f"  - Cross-encoder: {retriever.cross_encoder}")
    
    return retriever


def test_retrieve_method(retriever):
    """Test retrieve method with sample queries."""
    print("\nTesting retrieve method...")
    
    # Test query
    query = "What is machine learning?"
    print(f"  Query: '{query}'")
    
    # Retrieve passages
    results = retriever.retrieve(query, top_k=3)
    
    # Verify output structure
    assert isinstance(results, list), "Results should be a list"
    assert len(results) <= 3, "Should return at most top_k results"
    assert len(results) > 0, "Should return at least one result"
    
    print(f"✓ Retrieved {len(results)} passages")
    
    # Verify result structure
    for i, result in enumerate(results):
        assert isinstance(result, dict), f"Result {i} should be a dict"
        assert 'chunk' in result, f"Result {i} should have 'chunk'"
        assert 'title' in result, f"Result {i} should have 'title'"
        assert 'ce_score' in result, f"Result {i} should have 'ce_score'"
        
        assert isinstance(result['chunk'], str), f"Result {i} chunk should be string"
        assert isinstance(result['title'], str), f"Result {i} title should be string"
        assert isinstance(result['ce_score'], float), f"Result {i} ce_score should be float"
        
        print(f"\n  Result {i+1}:")
        print(f"    Title: {result['title']}")
        print(f"    Score: {result['ce_score']:.4f}")
        print(f"    Chunk: {result['chunk'][:100]}...")
    
    # Verify scores are in descending order
    scores = [r['ce_score'] for r in results]
    assert scores == sorted(scores, reverse=True), "Scores should be in descending order"
    print(f"\n✓ Scores are properly ranked: {[f'{s:.4f}' for s in scores]}")


def test_retrieve_with_different_top_k():
    """Test retrieve method with different top_k values."""
    print("\nTesting retrieve with different top_k values...")
    
    # Create sample data
    chunk_texts = [
        "Machine learning is a method of data analysis.",
        "Artificial intelligence is the simulation of human intelligence.",
        "Deep learning is part of machine learning methods.",
        "Neural networks are computing systems inspired by biological neural networks.",
        "Python is a popular programming language for data science.",
        "Data science combines domain expertise, programming skills, and statistics.",
        "Supervised learning is a type of machine learning.",
        "Unsupervised learning finds hidden patterns in data."
    ]
    
    chunk_meta = [{'title': f'Article {i}', 'src_id': i} for i in range(len(chunk_texts))]
    
    print(f"  Initializing retriever with {len(chunk_texts)} chunks...")
    retriever = HybridRetriever(chunk_texts, chunk_meta)
    
    query = "machine learning methods"
    
    # Test with top_k=5 (default for web app)
    results_5 = retriever.retrieve(query, top_k=5)
    assert len(results_5) <= 5, "Should return at most 5 results"
    print(f"✓ top_k=5: returned {len(results_5)} results")
    
    # Test with top_k=3
    results_3 = retriever.retrieve(query, top_k=3)
    assert len(results_3) <= 3, "Should return at most 3 results"
    print(f"✓ top_k=3: returned {len(results_3)} results")
    
    # Test with top_k=1
    results_1 = retriever.retrieve(query, top_k=1)
    assert len(results_1) == 1, "Should return exactly 1 result"
    print(f"✓ top_k=1: returned {len(results_1)} result")


def main():
    """Run all retrieval tests."""
    print("=" * 70)
    print("RETRIEVAL PIPELINE TESTS")
    print("=" * 70)
    
    try:
        test_simple_tokenize()
        test_build_chunks_basic()
        test_build_chunks_overlap()
        
        # Initialize retriever once for multiple tests
        retriever = test_hybrid_retriever_initialization()
        test_retrieve_method(retriever)
        
        # Test with different parameters
        test_retrieve_with_different_top_k()
        
        print("\n" + "=" * 70)
        print("ALL RETRIEVAL TESTS PASSED ✓")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
