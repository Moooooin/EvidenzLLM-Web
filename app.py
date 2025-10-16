"""
Flask backend for EvidenzLLM web chat interface.
Provides REST API for query processing with evidence-based answers.
"""

import os
import sys

# Load .env file FIRST
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system environment

# CRITICAL: Set environment variables BEFORE any imports
# This must happen before torch, transformers, or any ML library is imported
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from pipeline.pipeline import EvidenzPipeline


# Initialize Flask app with static folder
app = Flask(__name__, static_folder='static')

# Enable CORS for cross-origin requests
CORS(app)

# Global pipeline variable
pipeline = None


def init_pipeline():
    """
    Initialize pipeline on startup.
    Load configuration from environment variables and initialize EvidenzPipeline.
    """
    global pipeline
    
    # Load configuration from environment variables
    config = {
        'classifier_path': os.getenv('CLASSIFIER_PATH', './query_classifier_model'),
        'wiki_data_path': os.getenv('WIKI_DATA_PATH', './data/wiki_texts.pkl'),
        'gemini_api_key': os.getenv('GOOGLE_API_KEY'),
        'gemini_model': os.getenv('GEMINI_MODEL', 'gemini-1.5-pro')
    }
    
    # Validate that GOOGLE_API_KEY is present
    if not config['gemini_api_key']:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    
    # Add error handling and logging
    try:
        print("Initializing EvidenzLLM pipeline...")
        print(f"Classifier path: {config['classifier_path']}")
        print(f"Wiki data path: {config['wiki_data_path']}")
        print(f"Gemini model: {config['gemini_model']}")
        
        # Initialize EvidenzPipeline with configuration
        pipeline = EvidenzPipeline(config)
        
        print("Pipeline initialized successfully!")
        
    except FileNotFoundError as e:
        print(f"Error: Required file not found - {e}")
        raise
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        raise



@app.route('/')
def index():
    """
    Serve index.html from static folder.
    GET / endpoint to serve the chat interface.
    """
    return send_from_directory('static', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """
    Serve static files (CSS, JS, etc.).
    """
    return send_from_directory('static', path)


@app.route('/api/health', methods=['GET'])
def health():
    """
    Health check endpoint returning pipeline status.
    GET /api/health endpoint.
    
    Returns:
        JSON with status: 'ready' if pipeline is initialized, 'unavailable' otherwise
    """
    if pipeline is None:
        return jsonify({'status': 'unavailable'}), 503
    return jsonify({'status': 'ready'})


@app.route('/api/topics', methods=['GET'])
def topics():
    """
    Get available Wikipedia topics in the database.
    GET /api/topics endpoint.
    
    Returns:
        JSON with list of available topics
    """
    if pipeline is None:
        return jsonify({'error': 'Pipeline not initialized'}), 503
    
    try:
        available_topics = pipeline.get_available_topics()
        return jsonify({'topics': available_topics})
    except Exception as e:
        print(f"Error getting topics: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/query', methods=['POST'])
def query():
    """
    Main query endpoint.
    POST /api/query endpoint accepting JSON with question field.
    
    Request JSON:
        {
            "question": "Who discovered gravity?"
        }
    
    Response JSON:
        {
            "question": "Who discovered gravity?",
            "query_type": "factual_lookup",
            "answer": "Isaac Newton discovered gravity [1].",
            "passages": [
                {
                    "chunk": "Isaac Newton published his theory...",
                    "title": "Theory of Relativity",
                    "ce_score": 0.923
                }
            ]
        }
    
    Returns:
        JSON response with answer and evidence passages, or error message
    """
    try:
        # Validate request payload
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'Missing question field'}), 400
        
        # Validate question field
        question = data['question'].strip()
        if not question:
            return jsonify({'error': 'Empty question'}), 400
        
        print(f"Processing question: {question}")
        
        # Call pipeline.process_query
        result = pipeline.process_query(question)
        
        print(f"Successfully generated response")
        print(f"Result keys: {result.keys()}")
        print(f"Answer preview: {result.get('answer', 'NO ANSWER')[:100]}...")
        print(f"Query type: {result.get('query_type', 'NO TYPE')}")
        print(f"Passages count: {len(result.get('passages', []))}")
        
        # Fix NaN values in passages (NaN is not valid JSON)
        import math
        for passage in result.get('passages', []):
            if 'ce_score' in passage:
                score = passage['ce_score']
                if isinstance(score, float) and (math.isnan(score) or math.isinf(score)):
                    passage['ce_score'] = 0.0
        
        # Return JSON response
        return jsonify(result)
    
    except Exception as e:
        # Add error handling with appropriate HTTP status codes
        print(f"Error processing query: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    try:
        print("=" * 70)
        print("EvidenzLLM Web Chat - Starting Server")
        print("=" * 70)
        print()
        
        # Call init_pipeline on startup
        init_pipeline()
        
        # Read PORT from environment variable (default: 5000)
        port = int(os.getenv('PORT', 5000))
        
        print()
        print("=" * 70)
        print(f"Server starting on http://localhost:{port}")
        print("Press Ctrl+C to stop")
        print("=" * 70)
        print()
        
        # Run Flask app on 0.0.0.0 with specified port
        # use_reloader=False to avoid fork issues on macOS
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")
    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR STARTING SERVER")
        print("=" * 70)
        print(f"\nError: {e}")
        print("\nFor troubleshooting, see TROUBLESHOOTING.md")
        import traceback
        traceback.print_exc()
        sys.exit(1)
