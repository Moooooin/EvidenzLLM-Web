"""
Retrieval system module for EvidenzLLM.
Implements chunking, BM25, dense embeddings, and hybrid retrieval with cross-encoder reranking.
"""

import re
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import faiss
import torch
import os

# Ensure NLTK punkt tokenizer is available (lazy download)
def _ensure_nltk_data():
    """Ensure NLTK punkt tokenizer is downloaded."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        # Set download directory to avoid permission issues
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        nltk.download('punkt', quiet=True)


def build_chunks(wiki_entries, max_chars=1200, overlap_chars=200):
    """
    Teilt Wiki-Einträge in Text-Chunks auf.
    - max_chars: maximale Länge eines Chunks
    - overlap_chars: Anzahl überlappender Zeichen zwischen aufeinanderfolgenden Chunks

    Rückgabe:
    - chunks: Liste der Text-Chunks
    - meta: Metadaten zu jedem Chunk (Titel, Quelle)
    """
    # Ensure NLTK data is available
    _ensure_nltk_data()
    
    chunks, meta = [], []
    for idx, entry in enumerate(wiki_entries):
        # Mehrfach-Leerzeichen ersetzen und Text trimmen
        text = re.sub(r"\s+", " ", entry['text']).strip()

        # Text in Sätze zerlegen
        sents = sent_tokenize(text)
        buf = ""

        for s in sents:
            # Satz hinzufügen, falls max_chars nicht überschritten
            if len(buf) + len(s) <= max_chars:
                buf += (" " if buf else "") + s

            else:
                # Chunk speichern, wenn Buffer gefüllt
                if len(buf) > 0:
                    chunks.append(buf)
                    meta.append({"title": entry['title'], "src_id": idx})

                # Überlappende Zeichen behalten und neuen Satz hinzufügen
                buf_tail = buf[-overlap_chars:] if overlap_chars > 0 else ""
                buf = (buf_tail + " " + s).strip()

        # Letzten Buffer als Chunk speichern
        if len(buf) > 0:
            chunks.append(buf)
            meta.append({"title": entry['title'], "src_id": idx})

    return chunks, meta


def simple_tokenize(t):
    """
    Einfache Tokenisierungsfunktion.
    """
    t = t.lower()
    t = re.sub(r"[^a-z0-9äöüß ]+", " ", t)
    return t.split()



class HybridRetriever:
    """
    Combines BM25, dense embeddings, and cross-encoder reranking.
    Encapsulates the notebook's hybrid_retrieve function.
    """
    
    def __init__(self, chunk_texts, chunk_meta):
        """
        Initialize the hybrid retriever with BM25, dense embeddings, and cross-encoder.
        
        Args:
            chunk_texts: List of text chunks
            chunk_meta: List of metadata dicts for each chunk
        """
        self.chunk_texts = chunk_texts
        self.chunk_meta = chunk_meta
        
        # BM25 setup
        # Alle Chunks tokenisieren
        docs_tokens = [simple_tokenize(t) for t in chunk_texts]
        
        # BM25-Index für die tokenisierten Dokumente erstellen
        self.bm25 = BM25Okapi(docs_tokens)
        
        # Dense Embedding Modell laden (Sentence Transformer)
        self.dense_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
        
        # Embeddings für alle Chunks erstellen, normalisiert und als NumPy-Array
        emb_matrix = self.dense_model.encode(
            chunk_texts, convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        # FAISS-Index für inneres Produkt (Kosinus Ähnlichkeit) erstellen
        self.index_dense = faiss.IndexFlatIP(emb_matrix.shape[1])
        self.index_dense.add(emb_matrix)
        
        # Cross-Encoder laden
        self.cross_encoder = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

    
    def retrieve(self, query, top_k=5, alpha=0.6, bm25_candidates=200, dense_candidates=200, rerank_top=32):
        """
        Hybrides Retrieval aus Chunks:
        - Kombiniert BM25-Score und Dense Embeddings mit Gewicht alpha
        - Wählz zunächst Kandidaten aus BM25 und Dense Search
        - Rerankt Top-Kandidaten mit Cross-Encoder
        - Gibt die besten top_k Ergebnisse mit Text, Titel und Cross-Encoder Score zurück
        """

        # Query tokenisieren
        q_tokens = simple_tokenize(query)

        # BM25-Scores berechnen
        bm25_scores = self.bm25.get_scores(q_tokens)
        bm25_idx = np.argsort(bm25_scores)[-bm25_candidates:]

        # Dense Embedding für Query erzeugen und Index abfragen
        q_emb = self.dense_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index_dense.search(q_emb, dense_candidates)
        dense_scores = np.zeros(len(self.chunk_texts))
        dense_scores[I[0]] = D[0]

        # Hybrid-Score berechnen (gewichtete Mischung von BM25 und Dense Score)
        hybrid = alpha * dense_scores + (1 - alpha) * (bm25_scores / (np.max(bm25_scores) + 1e-9))

        # Kandidaten vereinigen (BM25 + Dense)
        cand_set = set(bm25_idx.tolist()) | set(I[0].tolist())
        cand_list = list(cand_set)
        cand_scores = [(i, hybrid[i]) for i in cand_list]

        # Top rerank_top nach Hybrid-Score auswählen
        cand_sorted = sorted(cand_scores, key=lambda x: x[1], reverse=True)[:rerank_top]

        # Query-Paare für Cross-Encoder vorbereiten
        pairs = [(query, self.chunk_texts[i]) for i, _ in cand_sorted]
        ce_scores = self.cross_encoder.predict(pairs)

        # Reranking nach Cross-Encoder Score
        reranked = sorted(zip([i for i,_ in cand_sorted], ce_scores), key=lambda x: x[1], reverse=True)

        # Top-K Ergebnisse auswählen
        top = reranked[:top_k]
        results = [{
            'chunk': self.chunk_texts[i],
            'title': self.chunk_meta[i]['title'],
            'ce_score': float(s)
        } for i, s in top]

        return results
