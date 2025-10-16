# Implementation Plan

- [x] 1. Set up project structure and dependencies
  - Create directory structure for backend modules (models, retrieval, generator, pipeline)
  - Create requirements.txt with all dependencies from notebook (torch, transformers, sentence-transformers, faiss-cpu, rank-bm25, google-generativeai, flask, flask-cors, nltk)
  - Create .env.example file with required environment variables
  - Create static directory for frontend files
  - _Requirements: 6.1, 6.2_

- [x] 2. Implement model classes module
  - [x] 2.1 Create models.py with QueryClassifier class
    - COPY EXACTLY: QueryClassifier class from notebook section "2. Modeldefinitionen" (lines with class QueryClassifier(nn.Module))
    - COPY EXACTLY: __init__ method with encoder, dropout, and classifier layers
    - COPY EXACTLY: forward method with CLS token extraction and loss calculation
    - COPY EXACTLY: label_map dictionary {"factual_lookup": 0, "explanation": 1, "reasoning": 2, "calculation": 3}
    - Only add: imports (torch, torch.nn, transformers.AutoModel)
    - _Requirements: 1.2, 1.3, 7.1_
  
  - [x] 2.2 Add model loading utilities
    - COPY EXACTLY: Model loading logic from notebook section "### Modell laden" (safetensors and pytorch_model.bin handling)
    - Wrap the loading code in a reusable function
    - Keep exact same error handling and file path checks
    - _Requirements: 1.5, 7.2, 7.5_

- [x] 3. Implement retrieval system module
  - [x] 3.1 Create retrieval.py with chunking functions
    - COPY EXACTLY: build_chunks function from notebook section "### Erstellen überlappender Text-Chunks" (entire function with max_chars=1200, overlap_chars=200)
    - COPY EXACTLY: simple_tokenize function from notebook section "### Tokenisierung und BM25 Initialisierung"
    - Add NLTK punkt download in initialization
    - Only add: necessary imports (nltk, re, sent_tokenize)
    - _Requirements: 2.1, 2.2_
  
  - [x] 3.2 Implement HybridRetriever class initialization
    - COPY EXACTLY: BM25 initialization code from notebook section "### Tokenisierung und BM25 Initialisierung" (docs_tokens and BM25Okapi setup)
    - COPY EXACTLY: Dense model loading from notebook section "### Dichter Vektorindex mit Cross-Encoder" (SentenceTransformer 'multi-qa-mpnet-base-dot-v1')
    - COPY EXACTLY: Embedding generation and FAISS index creation (emb_matrix, IndexFlatIP, normalize_embeddings=True)
    - COPY EXACTLY: Cross-encoder loading (CrossEncoder 'cross-encoder/ms-marco-MiniLM-L-6-v2')
    - Wrap this code in a class __init__ method, keep all parameters identical
    - _Requirements: 2.2, 2.3, 2.4_
  
  - [x] 3.3 Implement retrieve method in HybridRetriever
    - COPY EXACTLY: hybrid_retrieve function from notebook section "### Hybrides Retrieval" (entire function body)
    - Keep exact same variable names: q_tokens, bm25_scores, bm25_idx, q_emb, D, I, dense_scores, hybrid, cand_set, cand_list, cand_scores, cand_sorted, pairs, ce_scores, reranked, top, results
    - Keep exact same parameters: top_k=8 (change default to 5 for web app), alpha=0.6, bm25_candidates=200, dense_candidates=200, rerank_top=32
    - Keep exact same return format: list of dicts with 'chunk', 'title', 'ce_score'
    - Only change: wrap as class method instead of standalone function
    - _Requirements: 2.3, 2.4, 2.5, 2.6, 2.7_

- [x] 4. Implement answer generation module
  - [x] 4.1 Create generator.py with prompt building
    - COPY EXACTLY: RAG_SYSTEM constant from notebook section "### Generator laden und RAG Prompts erstellen"
    - COPY EXACTLY: FEW_SHOT_EXAMPLE constant from same notebook section
    - COPY EXACTLY: build_rag_prompt function from notebook section "### RAG Prompt zusammenstellen" (entire function with exact string formatting)
    - Keep exact same parameter names: question, passages, query_type
    - Keep exact same evidence formatting with [1], [2] numbering and Title: prefix
    - Only add: necessary imports (google.generativeai)
    - _Requirements: 3.1_
  
  - [x] 4.2 Implement GeminiGenerator class
    - Replace notebook's local Mistral model with Gemini API
    - Use genai.configure(api_key=api_key) for initialization
    - Use genai.GenerativeModel(model_name) to create model instance
    - In generate method: call model.generate_content(prompt) with generation_config
    - Set max_output_tokens parameter (equivalent to notebook's max_new_tokens=128)
    - Set temperature=0.1 for factual answers (low temperature like notebook's do_sample=False)
    - Extract response.text from API response
    - Add try/except for API errors with descriptive error messages
    - _Requirements: 3.2, 3.3, 3.4, 3.6, 3.7_

- [x] 5. Implement pipeline orchestrator
  - [x] 5.1 Create pipeline.py with EvidenzPipeline class initialization
    - COPY EXACTLY: device setup from notebook (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    - COPY EXACTLY: label_map dictionary from notebook section "## Query Classifier Datensatz"
    - Create reverse_label_map using {v:k for k,v in label_map.items()} pattern from notebook
    - COPY EXACTLY: tokenizer loading from notebook (AutoTokenizer.from_pretrained("microsoft/deberta-base"))
    - COPY EXACTLY: model initialization (QueryClassifier(num_labels=4))
    - Use the model loading utility from task 2.2 to load weights
    - COPY EXACTLY: model.to(device) and model.eval() calls from notebook
    - _Requirements: 1.1, 1.2, 7.1, 7.2, 8.1_
  
  - [x] 5.2 Add Wikipedia data loading and retriever initialization
    - COPY EXACTLY: pickle loading pattern from notebook (with open(..., 'rb') as f: pickle.load(f))
    - Call build_chunks function (already copied in task 3.1) on loaded wiki_texts
    - Initialize HybridRetriever (already implemented in task 3.2) with chunk_texts and chunk_meta
    - _Requirements: 2.1, 7.3, 7.4_
  
  - [x] 5.3 Implement classify_query method
    - COPY EXACTLY: Query classification code from notebook section "# 7. Demo" (lines starting with "qc_inputs = tokenizer_qc...")
    - COPY EXACTLY: tokenizer call with return_tensors='pt' and .to(device)
    - COPY EXACTLY: model input preparation {key: qc_inputs[key] for key in ['input_ids', 'attention_mask']}
    - COPY EXACTLY: model inference qc_out = model_qc(**qc_model_inputs)
    - COPY EXACTLY: prediction extraction torch.argmax(qc_out['logits'], dim=-1).item()
    - COPY EXACTLY: label mapping {v:k for k,v in label_map.items()}[qc_pred]
    - Wrap with torch.no_grad() for inference
    - _Requirements: 1.1, 1.3, 1.4_
  
  - [x] 5.4 Implement process_query method
    - COPY EXACTLY: Pipeline flow from notebook section "# 7. Demo"
    - Step 1: Call classify_query (mirrors notebook's query classification)
    - Step 2: Call retriever.retrieve with top_k=5 (mirrors notebook's hybrid_retrieve)
    - Step 3: Call build_rag_prompt (mirrors notebook's prompt building)
    - Step 4: Call generator.generate (replaces notebook's rag_model.generate with Gemini API)
    - Return dict with same structure as notebook demo output
    - _Requirements: 3.1, 3.4, 3.5_

- [x] 6. Implement Flask backend
  - [x] 6.1 Create app.py with Flask application setup
    - Initialize Flask app with static folder
    - Enable CORS for cross-origin requests
    - Create global pipeline variable
    - _Requirements: 5.1_
  
  - [x] 6.2 Implement init_pipeline function
    - Load configuration from environment variables (CLASSIFIER_PATH, WIKI_DATA_PATH, GOOGLE_API_KEY, GEMINI_MODEL)
    - Validate that GOOGLE_API_KEY is present
    - Initialize EvidenzPipeline with configuration
    - Add error handling and logging
    - _Requirements: 5.1, 6.2, 6.3, 6.4_
  
  - [x] 6.3 Implement API endpoints
    - Create GET / endpoint to serve index.html
    - Create GET /api/health endpoint returning pipeline status
    - Create POST /api/query endpoint accepting JSON with question field
    - Validate request payload and question field
    - Call pipeline.process_query and return JSON response
    - Add error handling with appropriate HTTP status codes
    - _Requirements: 5.2, 5.3, 5.4, 5.5, 5.6_
  
  - [x] 6.4 Add main entry point
    - Call init_pipeline on startup
    - Read PORT from environment variable (default: 5000)
    - Run Flask app on 0.0.0.0 with specified port
    - _Requirements: 5.1_

- [x] 7. Implement frontend HTML structure
  - Create static/index.html with chat interface layout
  - Add header with title "EvidenzLLM" and subtitle
  - Create chat-container div for message history
  - Create input-container with text input and send button
  - Add loading indicator div (initially hidden)
  - Link to style.css and app.js
  - _Requirements: 4.1, 4.2_

- [x] 8. Implement frontend styling
  - Create static/style.css with chat interface styles
  - Style container, header, and chat layout
  - Style user and assistant message bubbles differently
  - Style query type badge display
  - Style evidence passages with title, score, and text
  - Style input field and send button
  - Style loading indicator
  - Add responsive design for mobile devices
  - _Requirements: 4.1, 4.5, 4.6_

- [x] 9. Implement frontend JavaScript logic
  - [x] 9.1 Create static/app.js with DOM element references
    - Get references to messages div, input field, send button, and loading div
    - Define API_URL constant
    - _Requirements: 4.1_
  
  - [x] 9.2 Implement event listeners
    - Add click listener to send button calling sendQuery
    - Add keypress listener to input field for Enter key
    - _Requirements: 4.2, 4.3_
  
  - [x] 9.3 Implement sendQuery function
    - Get and validate question from input field
    - Call appendMessage to display user question
    - Clear input field
    - Show loading indicator and disable send button
    - Make POST request to /api/query with question
    - Handle response and call displayAnswer
    - Handle errors and display error message
    - Hide loading indicator and enable send button
    - _Requirements: 4.2, 4.3, 4.8_
  
  - [x] 9.4 Implement displayAnswer function
    - Create message div with assistant class
    - Add query type badge
    - Add answer text paragraph
    - Create evidence section with "Evidence:" header
    - Loop through passages and create passage divs
    - Display passage number, title, cross-encoder score
    - Display truncated passage text (first 200 chars)
    - Append to messages container and scroll to bottom
    - _Requirements: 4.4, 4.5, 4.6, 4.7_

- [x] 10. Create data preparation script
  - Create prepare_data.py script to generate Wikipedia data
  - COPY EXACTLY: Wikipedia fetching code from notebook section "### Wikipedia Artikel fetchen" (wikipediaapi.Wikipedia setup with user_agent)
  - COPY EXACTLY: topics list ["Machine Learning", "Artificial Intelligence", "Physics", "Theory of Relativity"]
  - COPY EXACTLY: Article fetching loop with wiki_wiki.page(topic), text chunking (chunk_size = 2000), and wiki_texts list building
  - COPY EXACTLY: Chunk filtering logic (len(chunk.strip())>50)
  - Save wiki_texts to data/wiki_texts.pkl using pickle.dump
  - Add command-line arguments for custom topics (optional enhancement)
  - _Requirements: 7.3, 7.4_

- [x] 11. Create configuration and documentation files
  - [x] 11.1 Create requirements.txt
    - List all Python dependencies with versions from notebook
    - Include torch, transformers, sentence-transformers, faiss-cpu, rank-bm25
    - Include google-generativeai, flask, flask-cors
    - Include nltk, numpy, safetensors
    - _Requirements: 6.1_
  
  - [x] 11.2 Create .env.example
    - Document GOOGLE_API_KEY variable
    - Document CLASSIFIER_PATH with default value
    - Document WIKI_DATA_PATH with default value
    - Document GEMINI_MODEL with default value
    - Document PORT with default value
    - _Requirements: 6.2, 6.3_
  
  - [x] 11.3 Create README.md
    - Add project description and features
    - Add installation instructions (pip install, data preparation)
    - Add configuration instructions (environment variables)
    - Add usage instructions (running server, accessing UI)
    - Add notes about model files from notebook
    - _Requirements: 6.1, 6.2, 7.1, 7.3_

- [x] 12. Test and validate implementation
  - [x] 12.1 Test model loading
    - Verify QueryClassifier loads from query_classifier_model directory
    - Test with both safetensors and pytorch_model.bin formats
    - Verify model outputs correct shape and format
    - _Requirements: 1.5, 7.1, 7.2, 7.5_
  
  - [x] 12.2 Test retrieval pipeline
    - Test build_chunks with sample Wikipedia text
    - Verify HybridRetriever initialization
    - Test retrieve method with sample queries
    - Verify 5 passages returned with correct structure
    - Compare scores with notebook outputs
    - _Requirements: 2.1, 2.7, 8.2_
  
  - [x] 12.3 Test Gemini API integration
    - Test GeminiGenerator with valid API key
    - Verify prompt format matches notebook
    - Test error handling with invalid API key
    - Verify timeout handling
    - _Requirements: 3.2, 3.3, 3.6, 3.7, 8.4_
  
  - [x] 12.4 Test full pipeline
    - Test EvidenzPipeline initialization
    - Test classify_query with questions from notebook
    - Test process_query end-to-end
    - Compare outputs with notebook demo results
    - Verify query types match notebook classifications
    - _Requirements: 1.3, 1.4, 8.2, 8.3_
  
  - [x] 12.5 Test Flask API
    - Test /api/health endpoint
    - Test /api/query with valid questions
    - Test error handling with invalid requests
    - Test concurrent request handling
    - Verify response format matches specification
    - _Requirements: 5.2, 5.3, 5.4, 5.5, 8.5_
  
  - [x] 12.6 Test frontend functionality
    - Test chat interface in browser
    - Verify message display for user and assistant
    - Verify evidence passages display correctly
    - Test loading indicator behavior
    - Test error message display
    - Test on different screen sizes
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8_
