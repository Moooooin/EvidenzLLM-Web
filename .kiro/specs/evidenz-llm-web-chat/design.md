# Design Document

## Overview

This design document outlines the architecture for transforming the EvidenzLLM Colab notebook into a web-based chat application. The system maintains the exact pipeline from the notebook: query classification using a trained DeBERTa model, hybrid retrieval (BM25 + dense embeddings + cross-encoder reranking), and answer generation via Google Gemini Pro API. The application consists of a Python Flask backend that hosts the ML pipeline and a simple HTML/JavaScript frontend for the chat interface.

## Architecture

### High-Level Architecture

```
┌─────────────────┐
│   Web Browser   │
│  (Chat UI)      │
└────────┬────────┘
         │ HTTP/JSON
         ▼
┌─────────────────┐
│  Flask Backend  │
│  - API Routes   │
│  - ML Pipeline  │
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────┐
    ▼         ▼          ▼          ▼
┌────────┐ ┌──────┐ ┌────────┐ ┌─────────┐
│Query   │ │Hybrid│ │Gemini  │ │Wikipedia│
│Classi- │ │Retri-│ │API     │ │Data     │
│fier    │ │eval  │ │        │ │         │
└────────┘ └──────┘ └────────┘ └─────────┘
```

### Technology Stack

- Backend: Python 3.8+, Flask
- ML Framework: PyTorch, Transformers (Hugging Face)
- Retrieval: sentence-transformers, FAISS, rank-bm25
- LLM: Google Gemini Pro API
- Frontend: HTML5, CSS3, Vanilla JavaScript
- Deployment: Can run locally or on cloud platforms

## Components and Interfaces

### 1. Model Classes (models.py)

Directly ported from the notebook with minimal changes.

#### QueryClassifier

```python
class QueryClassifier(nn.Module):
    """
    Query classifier using DeBERTa-base encoder.
    Identical to notebook implementation.
    """
    def __init__(self, num_labels=4, model_name="microsoft/deberta-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden, num_labels)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_emb)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}
```

Label mapping (from notebook):
```python
label_map = {
    "factual_lookup": 0,
    "explanation": 1,
    "reasoning": 2,
    "calculation": 3
}
```

### 2. Retrieval System (retrieval.py)

Implements the exact hybrid retrieval pipeline from the notebook.

#### Text Chunking

```python
def build_chunks(wiki_entries, max_chars=1200, overlap_chars=200):
    """
    Splits wiki entries into overlapping chunks.
    Identical to notebook implementation.
    """
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
```

#### Tokenization

```python
def simple_tokenize(t):
    """Simple tokenization for BM25. From notebook."""
    t = t.lower()
    t = re.sub(r"[^a-z0-9äöüß ]+", " ", t)
    return t.split()
```

#### HybridRetriever Class

```python
class HybridRetriever:
    """
    Combines BM25, dense embeddings, and cross-encoder reranking.
    Encapsulates the notebook's hybrid_retrieve function.
    """
    def __init__(self, chunk_texts, chunk_meta):
        # BM25 setup
        self.chunk_texts = chunk_texts
        self.chunk_meta = chunk_meta
        docs_tokens = [simple_tokenize(t) for t in chunk_texts]
        self.bm25 = BM25Okapi(docs_tokens)
        
        # Dense embedding setup
        self.dense_model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
        emb_matrix = self.dense_model.encode(
            chunk_texts, 
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        self.index_dense = faiss.IndexFlatIP(emb_matrix.shape[1])
        self.index_dense.add(emb_matrix)
        
        # Cross-encoder setup
        self.cross_encoder = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    def retrieve(self, query, top_k=5, alpha=0.6, 
                 bm25_candidates=200, dense_candidates=200, rerank_top=32):
        """
        Hybrid retrieval with exact notebook parameters.
        Returns list of dicts with 'chunk', 'title', 'ce_score'.
        """
        # BM25 scoring
        q_tokens = simple_tokenize(query)
        bm25_scores = self.bm25.get_scores(q_tokens)
        bm25_idx = np.argsort(bm25_scores)[-bm25_candidates:]
        
        # Dense scoring
        q_emb = self.dense_model.encode(
            [query], 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        D, I = self.index_dense.search(q_emb, dense_candidates)
        dense_scores = np.zeros(len(self.chunk_texts))
        dense_scores[I[0]] = D[0]
        
        # Hybrid score
        hybrid = alpha * dense_scores + (1 - alpha) * (
            bm25_scores / (np.max(bm25_scores) + 1e-9)
        )
        
        # Combine candidates
        cand_set = set(bm25_idx.tolist()) | set(I[0].tolist())
        cand_list = list(cand_set)
        cand_scores = [(i, hybrid[i]) for i in cand_list]
        cand_sorted = sorted(cand_scores, key=lambda x: x[1], reverse=True)[:rerank_top]
        
        # Cross-encoder reranking
        pairs = [(query, self.chunk_texts[i]) for i, _ in cand_sorted]
        ce_scores = self.cross_encoder.predict(pairs)
        reranked = sorted(
            zip([i for i,_ in cand_sorted], ce_scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Top-K results
        top = reranked[:top_k]
        results = [{
            'chunk': self.chunk_texts[i],
            'title': self.chunk_meta[i]['title'],
            'ce_score': float(s)
        } for i, s in top]
        
        return results
```

### 3. Answer Generation (generator.py)

Uses Google Gemini Pro API instead of local Mistral model.

#### Prompt Building

```python
# System prompt from notebook
RAG_SYSTEM = (
    "You are a precise assistant. Answer in English using ONLY the provided evidence snippets. "
    "Always give a direct answer followed by supporting citations in brackets."
)

# Few-shot example from notebook
FEW_SHOT_EXAMPLE = (
    "Example:\n"
    "Question: Who discovered gravity?\n"
    "Query Type: factual_lookup\n"
    "Evidence: [1] Title: Gravity\nIsaac Newton described universal gravitation...\n"
    "Answer: Isaac Newton discovered gravity [1].\n"
)

def build_rag_prompt(question, passages, query_type):
    """
    Builds RAG prompt exactly as in notebook.
    """
    evidence = "\n\n".join([
        f"[{i+1}] Title: {p['title']}\n{p['chunk']}" 
        for i, p in enumerate(passages)
    ])
    
    prompt = (
        f"{RAG_SYSTEM}\n\n"
        f"{FEW_SHOT_EXAMPLE}\n"
        f"Question: {question}\n\n"
        f"Query Type: {query_type}\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Answer:"
    )
    
    return prompt
```

#### Gemini API Integration

```python
import google.generativeai as genai

class GeminiGenerator:
    """
    Replaces local Mistral model with Gemini API.
    """
    def __init__(self, api_key, model_name="gemini-1.5-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def generate(self, prompt, max_tokens=256):
        """
        Generates answer using Gemini API.
        """
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.1,  # Low temperature for factual answers
                )
            )
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")
```

### 4. Pipeline Orchestrator (pipeline.py)

Coordinates all components, mirroring the notebook's demo section.

```python
class EvidenzPipeline:
    """
    Main pipeline orchestrator.
    Replicates the end-to-end demo from notebook.
    """
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_map = {
            "factual_lookup": 0,
            "explanation": 1,
            "reasoning": 2,
            "calculation": 3
        }
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        # Load query classifier
        self.qc_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
        self.qc_model = QueryClassifier(num_labels=4)
        self._load_classifier_weights(config['classifier_path'])
        self.qc_model.to(self.device)
        self.qc_model.eval()
        
        # Load Wikipedia data and build retriever
        wiki_texts = self._load_wikipedia_data(config['wiki_data_path'])
        chunk_texts, chunk_meta = build_chunks(wiki_texts)
        self.retriever = HybridRetriever(chunk_texts, chunk_meta)
        
        # Initialize Gemini generator
        self.generator = GeminiGenerator(
            api_key=config['gemini_api_key'],
            model_name=config.get('gemini_model', 'gemini-1.5-pro')
        )
    
    def _load_classifier_weights(self, path):
        """Load model weights from safetensors or pytorch_model.bin"""
        safetensors_path = os.path.join(path, 'model.safetensors')
        pytorch_path = os.path.join(path, 'pytorch_model.bin')
        
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path)
        elif os.path.exists(pytorch_path):
            state_dict = torch.load(pytorch_path, map_location=self.device)
        else:
            raise FileNotFoundError(f"No model file found in {path}")
        
        self.qc_model.load_state_dict(state_dict)
    
    def _load_wikipedia_data(self, path):
        """Load Wikipedia texts from pickle file"""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def classify_query(self, question):
        """
        Classify query type. From notebook demo.
        """
        inputs = self.qc_tokenizer(question, return_tensors='pt').to(self.device)
        model_inputs = {
            key: inputs[key] 
            for key in ['input_ids', 'attention_mask']
        }
        
        with torch.no_grad():
            outputs = self.qc_model(**model_inputs)
            pred = torch.argmax(outputs['logits'], dim=-1).item()
        
        return self.reverse_label_map[pred]
    
    def process_query(self, question):
        """
        Full pipeline: classify -> retrieve -> generate.
        Mirrors notebook's demo section.
        """
        # Step 1: Classify query
        query_type = self.classify_query(question)
        
        # Step 2: Retrieve evidence
        passages = self.retriever.retrieve(question, top_k=5)
        
        # Step 3: Build prompt
        prompt = build_rag_prompt(question, passages, query_type)
        
        # Step 4: Generate answer
        answer = self.generator.generate(prompt, max_tokens=256)
        
        return {
            'question': question,
            'query_type': query_type,
            'answer': answer,
            'passages': passages
        }
```

### 5. Flask Backend (app.py)

Simple REST API exposing the pipeline.

```python
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__, static_folder='static')
CORS(app)

# Global pipeline instance
pipeline = None

def init_pipeline():
    """Initialize pipeline on startup"""
    global pipeline
    config = {
        'classifier_path': os.getenv('CLASSIFIER_PATH', './query_classifier_model'),
        'wiki_data_path': os.getenv('WIKI_DATA_PATH', './data/wiki_texts.pkl'),
        'gemini_api_key': os.getenv('GOOGLE_API_KEY'),
        'gemini_model': os.getenv('GEMINI_MODEL', 'gemini-1.5-pro')
    }
    
    if not config['gemini_api_key']:
        raise ValueError("GOOGLE_API_KEY environment variable required")
    
    pipeline = EvidenzPipeline(config)
    print("Pipeline initialized successfully")

@app.route('/')
def index():
    """Serve frontend"""
    return send_from_directory('static', 'index.html')

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if pipeline is None:
        return jsonify({'status': 'unavailable'}), 503
    return jsonify({'status': 'ready'})

@app.route('/api/query', methods=['POST'])
def query():
    """
    Main query endpoint.
    Request: {"question": "Who discovered gravity?"}
    Response: {
        "question": "...",
        "query_type": "factual_lookup",
        "answer": "...",
        "passages": [...]
    }
    """
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'Missing question field'}), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({'error': 'Empty question'}), 400
        
        result = pipeline.process_query(question)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_pipeline()
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
```

### 6. Frontend (static/index.html, static/style.css, static/app.js)

Simple chat interface with message history.

#### HTML Structure

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EvidenzLLM Chat</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>EvidenzLLM</h1>
            <p>Evidence-based Question Answering</p>
        </header>
        
        <div id="chat-container">
            <div id="messages"></div>
        </div>
        
        <div id="input-container">
            <input 
                type="text" 
                id="question-input" 
                placeholder="Ask a question..."
                autocomplete="off"
            >
            <button id="send-btn">Send</button>
        </div>
        
        <div id="loading" class="hidden">Processing...</div>
    </div>
    
    <script src="app.js"></script>
</body>
</html>
```

#### JavaScript Logic

```javascript
const API_URL = '/api/query';
const messagesDiv = document.getElementById('messages');
const inputField = document.getElementById('question-input');
const sendBtn = document.getElementById('send-btn');
const loadingDiv = document.getElementById('loading');

// Send query on button click or Enter key
sendBtn.addEventListener('click', sendQuery);
inputField.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendQuery();
});

async function sendQuery() {
    const question = inputField.value.trim();
    if (!question) return;
    
    // Display user message
    appendMessage('user', question);
    inputField.value = '';
    
    // Show loading
    loadingDiv.classList.remove('hidden');
    sendBtn.disabled = true;
    
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({question})
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Request failed');
        }
        
        const data = await response.json();
        displayAnswer(data);
        
    } catch (error) {
        appendMessage('error', `Error: ${error.message}`);
    } finally {
        loadingDiv.classList.add('hidden');
        sendBtn.disabled = false;
    }
}

function appendMessage(type, content) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${type}`;
    msgDiv.textContent = content;
    messagesDiv.appendChild(msgDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function displayAnswer(data) {
    // Answer message
    const answerDiv = document.createElement('div');
    answerDiv.className = 'message assistant';
    
    const queryTypeSpan = document.createElement('span');
    queryTypeSpan.className = 'query-type';
    queryTypeSpan.textContent = `[${data.query_type}]`;
    
    const answerText = document.createElement('p');
    answerText.textContent = data.answer;
    
    answerDiv.appendChild(queryTypeSpan);
    answerDiv.appendChild(answerText);
    
    // Evidence passages
    const evidenceDiv = document.createElement('div');
    evidenceDiv.className = 'evidence';
    evidenceDiv.innerHTML = '<strong>Evidence:</strong>';
    
    data.passages.forEach((passage, idx) => {
        const passageDiv = document.createElement('div');
        passageDiv.className = 'passage';
        passageDiv.innerHTML = `
            <div class="passage-header">
                [${idx + 1}] ${passage.title} 
                <span class="score">(score: ${passage.ce_score.toFixed(3)})</span>
            </div>
            <div class="passage-text">${passage.chunk.substring(0, 200)}...</div>
        `;
        evidenceDiv.appendChild(passageDiv);
    });
    
    answerDiv.appendChild(evidenceDiv);
    messagesDiv.appendChild(answerDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}
```

## Data Models

### Configuration

```python
{
    'classifier_path': str,      # Path to query_classifier_model directory
    'wiki_data_path': str,        # Path to wiki_texts.pkl
    'gemini_api_key': str,        # Google API key
    'gemini_model': str           # Model name (default: gemini-1.5-pro)
}
```

### API Request/Response

Request:
```json
{
    "question": "Who developed the theory of relativity?"
}
```

Response:
```json
{
    "question": "Who developed the theory of relativity?",
    "query_type": "factual_lookup",
    "answer": "Albert Einstein developed the theory of relativity [1][2].",
    "passages": [
        {
            "chunk": "Albert Einstein published his theory...",
            "title": "Theory of Relativity",
            "ce_score": 0.923
        }
    ]
}
```

## Error Handling

### Backend Errors

1. Model Loading Failures
   - Check file paths and formats
   - Log detailed error messages
   - Return 503 Service Unavailable

2. API Errors
   - Catch Gemini API exceptions
   - Return 500 with error details
   - Log for debugging

3. Invalid Requests
   - Validate JSON payload
   - Check required fields
   - Return 400 Bad Request

### Frontend Errors

1. Network Errors
   - Display user-friendly message
   - Allow retry

2. Empty Responses
   - Handle gracefully
   - Show appropriate message

## Testing Strategy

### Unit Tests

1. Model Loading
   - Test safetensors and pytorch_model.bin loading
   - Verify model outputs match expected format

2. Retrieval Components
   - Test chunking with various text lengths
   - Verify BM25, dense, and cross-encoder scores
   - Test hybrid scoring calculation

3. Prompt Building
   - Verify exact format matches notebook
   - Test with different query types

### Integration Tests

1. Full Pipeline
   - Test with sample questions from notebook
   - Verify query classification accuracy
   - Check retrieval returns 5 passages
   - Validate Gemini API integration

2. API Endpoints
   - Test /api/health
   - Test /api/query with valid/invalid inputs
   - Test error handling

### Manual Testing

1. Compare outputs with notebook
   - Use same questions
   - Verify query types match
   - Check passage relevance
   - Compare answer quality

2. UI Testing
   - Test chat flow
   - Verify evidence display
   - Check responsive behavior

## Deployment Considerations

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GOOGLE_API_KEY="your-key"
export CLASSIFIER_PATH="./query_classifier_model"
export WIKI_DATA_PATH="./data/wiki_texts.pkl"

# Run server
python app.py
```

### Production Deployment

1. Use production WSGI server (gunicorn)
2. Set appropriate environment variables
3. Configure CORS for specific domains
4. Enable HTTPS
5. Add request rate limiting
6. Monitor API usage and costs

### Resource Requirements

- RAM: 4GB minimum (models + indices)
- GPU: Optional, improves classification speed
- Storage: ~2GB for models and Wikipedia data
- Network: Stable connection for Gemini API

## Performance Optimization

1. Model Loading
   - Load models once at startup
   - Keep in memory for fast inference

2. Retrieval Caching
   - Cache FAISS index in memory
   - Precompute BM25 structures

3. API Calls
   - Set reasonable timeouts
   - Implement retry logic with backoff

4. Frontend
   - Minimize JavaScript bundle
   - Use CSS for smooth animations
   - Debounce input if needed
