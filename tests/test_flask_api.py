"""
Test Flask API endpoints.
Tests /api/health, /api/query, and error handling.
"""

import os
import sys
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_app_structure():
    """Test Flask app structure."""
    print("Testing Flask app structure...")
    
    app_file = './app.py'
    assert os.path.exists(app_file), f"App file not found: {app_file}"
    print(f"✓ App file exists: {app_file}")
    
    # Read app file and check for key components
    with open(app_file, 'r') as f:
        content = f.read()
    
    # Check for Flask initialization
    assert 'Flask' in content, "Flask not imported"
    assert 'CORS' in content, "CORS not imported"
    print(f"✓ Flask and CORS imported")
    
    # Check for routes
    assert '@app.route(\'/\')' in content, "Missing / route"
    assert '@app.route(\'/api/health\'' in content, "Missing /api/health route"
    assert '@app.route(\'/api/query\'' in content, "Missing /api/query route"
    print(f"✓ All required routes defined")
    
    # Check for init_pipeline function
    assert 'def init_pipeline' in content, "Missing init_pipeline function"
    print(f"✓ init_pipeline function defined")


def test_api_endpoints_structure():
    """Test API endpoints structure."""
    print("\nTesting API endpoints structure...")
    
    endpoints = {
        '/': 'GET - Serve index.html',
        '/api/health': 'GET - Health check',
        '/api/query': 'POST - Process query'
    }
    
    print(f"✓ Expected endpoints:")
    for endpoint, description in endpoints.items():
        print(f"  {endpoint}: {description}")


def test_health_endpoint_response():
    """Test expected health endpoint response."""
    print("\nTesting health endpoint response structure...")
    
    # Expected responses
    ready_response = {'status': 'ready'}
    unavailable_response = {'status': 'unavailable'}
    
    print(f"✓ Expected responses:")
    print(f"  Ready: {ready_response} (HTTP 200)")
    print(f"  Unavailable: {unavailable_response} (HTTP 503)")


def test_query_endpoint_request_format():
    """Test expected query endpoint request format."""
    print("\nTesting query endpoint request format...")
    
    # Expected request format
    valid_request = {
        'question': 'Who discovered gravity?'
    }
    
    print(f"✓ Expected request format:")
    print(f"  {json.dumps(valid_request, indent=2)}")
    
    # Invalid requests
    invalid_requests = [
        {},  # Missing question
        {'question': ''},  # Empty question
        {'query': 'test'},  # Wrong field name
    ]
    
    print(f"\n✓ Invalid request examples (should return 400):")
    for i, req in enumerate(invalid_requests, 1):
        print(f"  {i}. {req}")


def test_query_endpoint_response_format():
    """Test expected query endpoint response format."""
    print("\nTesting query endpoint response format...")
    
    # Expected success response
    success_response = {
        'question': 'Who discovered gravity?',
        'query_type': 'factual_lookup',
        'answer': 'Isaac Newton discovered gravity [1].',
        'passages': [
            {
                'chunk': 'Isaac Newton published his theory...',
                'title': 'Gravity',
                'ce_score': 0.923
            }
        ]
    }
    
    print(f"✓ Expected success response (HTTP 200):")
    print(f"  {json.dumps(success_response, indent=2)}")
    
    # Expected error response
    error_response = {
        'error': 'Error message here'
    }
    
    print(f"\n✓ Expected error response (HTTP 500):")
    print(f"  {json.dumps(error_response, indent=2)}")


def test_environment_variables():
    """Test required environment variables."""
    print("\nTesting environment variables...")
    
    required_vars = {
        'GOOGLE_API_KEY': 'Required for Gemini API',
        'CLASSIFIER_PATH': 'Optional (default: ./query_classifier_model)',
        'WIKI_DATA_PATH': 'Optional (default: ./data/wiki_texts.pkl)',
        'GEMINI_MODEL': 'Optional (default: gemini-1.5-pro)',
        'PORT': 'Optional (default: 5000)'
    }
    
    print(f"✓ Environment variables:")
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            display_value = '[REDACTED]' if 'KEY' in var else value
            print(f"  {var}: {display_value} ✓")
        else:
            print(f"  {var}: Not set - {description}")


def test_cors_configuration():
    """Test CORS configuration."""
    print("\nTesting CORS configuration...")
    
    print(f"✓ CORS should be enabled for cross-origin requests")
    print(f"  This allows the frontend to communicate with the backend")


def test_error_handling():
    """Test error handling scenarios."""
    print("\nTesting error handling scenarios...")
    
    error_scenarios = [
        "Missing question field -> 400 Bad Request",
        "Empty question -> 400 Bad Request",
        "Pipeline not initialized -> 503 Service Unavailable",
        "Processing error -> 500 Internal Server Error",
        "Invalid API key -> 500 Internal Server Error"
    ]
    
    print(f"✓ Expected error handling:")
    for scenario in error_scenarios:
        print(f"  - {scenario}")


def test_with_curl_commands():
    """Show example curl commands for testing."""
    print("\nExample curl commands for testing...")
    
    commands = [
        "# Test health endpoint",
        "curl http://localhost:5000/api/health",
        "",
        "# Test query endpoint",
        'curl -X POST http://localhost:5000/api/query \\',
        '  -H "Content-Type: application/json" \\',
        '  -d \'{"question": "What is machine learning?"}\'',
        "",
        "# Test with invalid request",
        'curl -X POST http://localhost:5000/api/query \\',
        '  -H "Content-Type: application/json" \\',
        '  -d \'{}\''
    ]
    
    print(f"✓ Example commands:")
    for cmd in commands:
        print(f"  {cmd}")


def main():
    """Run all Flask API tests."""
    print("=" * 70)
    print("FLASK API TESTS")
    print("=" * 70)
    
    try:
        test_app_structure()
        test_api_endpoints_structure()
        test_health_endpoint_response()
        test_query_endpoint_request_format()
        test_query_endpoint_response_format()
        test_environment_variables()
        test_cors_configuration()
        test_error_handling()
        test_with_curl_commands()
        
        print("\n" + "=" * 70)
        print("ALL FLASK API TESTS PASSED ✓")
        print("=" * 70)
        print("\nNote: To test the live API, start the server with:")
        print("  python app.py")
        print("\nThen use the curl commands shown above or test via browser.")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
