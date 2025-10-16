"""
Minimal Flask app for testing without heavy ML models.
Use this to verify the web interface works before loading models.
"""

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

app = Flask(__name__, static_folder='static')
CORS(app)

@app.route('/')
def index():
    """Serve the chat interface."""
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

@app.route('/api/health', methods=['GET'])
def health():
    """Health check."""
    return jsonify({'status': 'ready'})

@app.route('/api/query', methods=['POST'])
def query():
    """Mock query endpoint for testing."""
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'Missing question field'}), 400
        
        question = data['question'].strip()
        if not question:
            return jsonify({'error': 'Empty question'}), 400
        
        # Return mock response
        result = {
            'question': question,
            'query_type': 'explanation',
            'answer': f'This is a mock response to: "{question}". The full ML pipeline is not loaded in minimal mode. To use the real system, you need to fix the threading issues first.',
            'passages': [
                {
                    'chunk': 'This is a mock evidence passage. In the full system, this would be retrieved from Wikipedia using hybrid retrieval.',
                    'title': 'Mock Article',
                    'ce_score': 0.95
                },
                {
                    'chunk': 'Another mock passage showing how evidence would be displayed.',
                    'title': 'Test Document',
                    'ce_score': 0.87
                }
            ]
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    
    print("=" * 70)
    print("EvidenzLLM Web Chat - MINIMAL MODE")
    print("=" * 70)
    print()
    print("This is a minimal version without ML models.")
    print("Use this to test the web interface.")
    print()
    print(f"Server starting on http://localhost:{port}")
    print("Press Ctrl+C to stop")
    print("=" * 70)
    print()
    
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
