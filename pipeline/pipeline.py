"""
Pipeline orchestrator for EvidenzLLM.
Coordinates query classification, retrieval, and answer generation.
"""

import os
import pickle

# Lazy imports to avoid threading issues on macOS
# These will be imported when EvidenzPipeline is instantiated
_torch = None
_AutoTokenizer = None
_QueryClassifier = None
_load_query_classifier = None
_build_chunks = None
_HybridRetriever = None
_GeminiGenerator = None
_build_rag_prompt = None

def _ensure_imports():
    """Lazy import of heavy dependencies."""
    global _torch, _AutoTokenizer, _QueryClassifier, _load_query_classifier
    global _build_chunks, _HybridRetriever, _GeminiGenerator, _build_rag_prompt
    
    if _torch is None:
        import torch as _torch_module
        _torch = _torch_module
        
        from transformers import AutoTokenizer as _AT
        _AutoTokenizer = _AT
        
        from models.models import QueryClassifier as _QC, load_query_classifier as _LQC
        _QueryClassifier = _QC
        _load_query_classifier = _LQC
        
        from retrieval.retrieval import build_chunks as _BC, HybridRetriever as _HR
        _build_chunks = _BC
        _HybridRetriever = _HR
        
        from generator.generator import GeminiGenerator as _GG, build_rag_prompt as _BRP
        _GeminiGenerator = _GG
        _build_rag_prompt = _BRP
    
    return (_torch, _AutoTokenizer, _QueryClassifier, _load_query_classifier,
            _build_chunks, _HybridRetriever, _GeminiGenerator, _build_rag_prompt)


class EvidenzPipeline:
    """
    Main pipeline orchestrator.
    Replicates the end-to-end demo from notebook.
    """
    
    def __init__(self, config):
        """
        Initialize the EvidenzLLM pipeline.
        
        Args:
            config: Dictionary with configuration:
                - classifier_path: Path to query classifier model directory
                - wiki_data_path: Path to Wikipedia data pickle file
                - gemini_api_key: Google API key for Gemini
                - gemini_model: Gemini model name (optional, default: gemini-1.5-pro)
        """
        # Ensure imports are loaded
        (torch, AutoTokenizer, QueryClassifier, load_query_classifier,
         build_chunks, HybridRetriever, GeminiGenerator, build_rag_prompt) = _ensure_imports()
        
        # Store references for use in methods
        self._torch = torch
        self._AutoTokenizer = AutoTokenizer
        self._load_query_classifier = load_query_classifier
        self._build_chunks = build_chunks
        self._HybridRetriever = HybridRetriever
        self._GeminiGenerator = GeminiGenerator
        self._build_rag_prompt = build_rag_prompt
        
        # Device setup from notebook
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Label map from notebook section "## Query Classifier Datensatz"
        self.label_map = {
            "factual_lookup": 0,
            "explanation": 1,
            "reasoning": 2,
            "calculation": 3
        }
        
        # Create reverse label map using pattern from notebook
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        # Load tokenizer from notebook (AutoTokenizer.from_pretrained("microsoft/deberta-base"))
        print("Loading tokenizer...")
        self.qc_tokenizer = self._AutoTokenizer.from_pretrained("microsoft/deberta-base")
        
        # Load query classifier model
        print("Loading query classifier model...")
        self.qc_model = self._load_query_classifier(config['classifier_path'], self.device)
        
        # Model is already in eval mode from load_query_classifier, but ensure it
        self.qc_model.to(self.device)
        self.qc_model.eval()
        
        # Load Wikipedia data and initialize retriever (will be done in next subtask)
        print("Loading Wikipedia data...")
        wiki_texts = self._load_wikipedia_data(config['wiki_data_path'])
        
        print("Building chunks...")
        chunk_texts, chunk_meta = self._build_chunks(wiki_texts)
        
        print("Initializing hybrid retriever...")
        self.retriever = self._HybridRetriever(chunk_texts, chunk_meta)
        
        # Extract unique topics from chunk metadata
        print("Extracting available topics...")
        self.available_topics = self._extract_topics(chunk_meta)
        print(f"Found {len(self.available_topics)} unique topics")
        
        # Initialize Gemini generator
        print("Initializing Gemini generator...")
        self.generator = self._GeminiGenerator(
            api_key=config['gemini_api_key'],
            model_name=config.get('gemini_model', 'gemini-1.5-pro')
        )
        
        print("Pipeline initialization complete!")
    
    def _load_wikipedia_data(self, path):
        """
        Load Wikipedia texts from pickle file.
        Copied exactly from notebook pickle loading pattern.
        
        Args:
            path: Path to wiki_texts.pkl file
            
        Returns:
            List of Wikipedia text entries
        """
        with open(path, 'rb') as f:
            wiki_texts = pickle.load(f)
        return wiki_texts
    
    def _extract_topics(self, chunk_meta):
        """
        Extract unique Wikipedia topics from chunk metadata.
        
        Args:
            chunk_meta: List of metadata dictionaries for each chunk
            
        Returns:
            Sorted list of unique topic titles
        """
        topics = set()
        for meta in chunk_meta:
            if 'title' in meta and meta['title']:
                topics.add(meta['title'])
        return sorted(list(topics))
    
    def get_available_topics(self):
        """
        Get list of available Wikipedia topics in the database.
        
        Returns:
            List of topic titles
        """
        return self.available_topics
    
    def classify_query(self, question):
        """
        Classify query type.
        Copied exactly from notebook section "# 7. Demo".
        
        Args:
            question: The user's question string
            
        Returns:
            Query type label (factual_lookup, explanation, reasoning, or calculation)
        """
        # Tokenizer call with return_tensors='pt' and .to(device)
        qc_inputs = self.qc_tokenizer(question, return_tensors='pt').to(self.device)
        
        # Model input preparation {key: qc_inputs[key] for key in ['input_ids', 'attention_mask']}
        qc_model_inputs = {key: qc_inputs[key] for key in ['input_ids', 'attention_mask']}
        
        # Wrap with torch.no_grad() for inference
        with self._torch.no_grad():
            # Model inference qc_out = model_qc(**qc_model_inputs)
            qc_out = self.qc_model(**qc_model_inputs)
            
            # Prediction extraction torch.argmax(qc_out['logits'], dim=-1).item()
            qc_pred = self._torch.argmax(qc_out['logits'], dim=-1).item()
        
        # Label mapping {v:k for k,v in label_map.items()}[qc_pred]
        query_type = self.reverse_label_map[qc_pred]
        
        return query_type
    
    def process_query(self, question):
        """
        Full pipeline: classify -> retrieve -> generate.
        Mirrors notebook's demo section "# 7. Demo".
        
        Args:
            question: The user's question string
            
        Returns:
            Dictionary with:
                - question: Original question
                - query_type: Classified query type
                - answer: Generated answer
                - passages: List of evidence passages
        """
        # Step 1: Call classify_query (mirrors notebook's query classification)
        query_type = self.classify_query(question)
        
        # Step 2: Call retriever.retrieve with top_k=5 (mirrors notebook's hybrid_retrieve)
        passages = self.retriever.retrieve(question, top_k=5)
        
        # Step 3: Call build_rag_prompt (mirrors notebook's prompt building)
        prompt = self._build_rag_prompt(question, passages, query_type)
        
        # Step 4: Call generator.generate (replaces notebook's rag_model.generate with Gemini API)
        answer = self.generator.generate(prompt, max_tokens=512)
        
        # Return dict with same structure as notebook demo output
        return {
            'question': question,
            'query_type': query_type,
            'answer': answer,
            'passages': passages
        }
