"""
Manual test of retrieval functions without full module import.
"""

import re
from nltk.tokenize import sent_tokenize


def simple_tokenize(t):
    """Simple tokenization for BM25."""
    t = t.lower()
    t = re.sub(r"[^a-z0-9äöüß ]+", " ", t)
    return t.split()


def build_chunks(wiki_entries, max_chars=1200, overlap_chars=200):
    """Build overlapping text chunks from wiki entries."""
    chunks, meta = [], []
    for idx, entry in enumerate(wiki_entries):
        text = re.sub(r"\s+", " ", entry['text']).strip()
        sents = sent_tokenize(text)
        buf = ""
        
        for s in sents:
            if len(buf) + len(s) <= max_chars:
                buf += (" " if buf else "") + s
            else:
                if len(buf) > 0:
                    chunks.append(buf)
                    meta.append({"title": entry['title'], "src_id": idx})
                buf_tail = buf[-overlap_chars:] if overlap_chars > 0 else ""
                buf = (buf_tail + " " + s).strip()
        
        if len(buf) > 0:
            chunks.append(buf)
            meta.append({"title": entry['title'], "src_id": idx})
    
    return chunks, meta


def test_simple_tokenize():
    """Test simple tokenization function."""
    print("Testing simple_tokenize...")
    
    text = "Hello World! This is a TEST."
    tokens = simple_tokenize(text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    print(f"✓ Tokenized '{text}' -> {tokens}")
    
    text2 = "UPPERCASE lowercase"
    tokens2 = simple_tokenize(text2)
    assert all(t.islower() or t.isdigit() for t in tokens2)
    print(f"✓ Lowercase conversion works: {tokens2}")


def test_build_chunks():
    """Test build_chunks function."""
    print("\nTesting build_chunks...")
    
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
    
    chunks, meta = build_chunks(wiki_entries, max_chars=200, overlap_chars=50)
    
    assert isinstance(chunks, list)
    assert isinstance(meta, list)
    assert len(chunks) == len(meta)
    assert len(chunks) > 0
    
    print(f"✓ Created {len(chunks)} chunks from {len(wiki_entries)} entries")
    
    for i, (chunk, m) in enumerate(zip(chunks[:3], meta[:3])):
        assert isinstance(chunk, str)
        assert len(chunk) > 0
        assert 'title' in m
        assert 'src_id' in m
        print(f"  Chunk {i}: {len(chunk)} chars, title='{m['title']}'")
    
    print(f"✓ All chunks have correct structure")


def test_build_chunks_default_params():
    """Test with default notebook parameters."""
    print("\nTesting with default parameters (1200, 200)...")
    
    long_text = "This is a test sentence. " * 100
    wiki_entries = [{'title': 'Long Article', 'text': long_text}]
    
    chunks, meta = build_chunks(wiki_entries, max_chars=1200, overlap_chars=200)
    
    print(f"✓ Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):
        print(f"  Chunk {i}: {len(chunk)} chars")


def main():
    """Run manual retrieval tests."""
    print("=" * 70)
    print("MANUAL RETRIEVAL TESTS")
    print("=" * 70)
    
    try:
        test_simple_tokenize()
        test_build_chunks()
        test_build_chunks_default_params()
        
        print("\n" + "=" * 70)
        print("ALL MANUAL RETRIEVAL TESTS PASSED ✓")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
