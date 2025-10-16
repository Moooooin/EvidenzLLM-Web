"""
Test frontend functionality.
Tests HTML structure, CSS styling, and JavaScript logic.
"""

import os
import sys
import re

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_html_structure():
    """Test HTML file structure."""
    print("Testing HTML structure...")
    
    html_file = './static/index.html'
    assert os.path.exists(html_file), f"HTML file not found: {html_file}"
    print(f"✓ HTML file exists: {html_file}")
    
    with open(html_file, 'r') as f:
        content = f.read()
    
    # Check DOCTYPE and basic structure
    assert '<!DOCTYPE html>' in content, "Missing DOCTYPE declaration"
    assert '<html' in content, "Missing html tag"
    assert '<head>' in content, "Missing head tag"
    assert '<body>' in content, "Missing body tag"
    print(f"✓ Basic HTML structure present")
    
    # Check meta tags
    assert 'charset="UTF-8"' in content, "Missing UTF-8 charset"
    assert 'viewport' in content, "Missing viewport meta tag"
    print(f"✓ Meta tags present")
    
    # Check title
    assert '<title>EvidenzLLM Chat</title>' in content, "Missing or incorrect title"
    print(f"✓ Title correct: 'EvidenzLLM Chat'")
    
    # Check CSS link
    assert 'href="style.css"' in content, "Missing style.css link"
    print(f"✓ CSS stylesheet linked")
    
    # Check JavaScript link
    assert 'src="app.js"' in content, "Missing app.js script"
    print(f"✓ JavaScript file linked")


def test_html_elements():
    """Test required HTML elements."""
    print("\nTesting HTML elements...")
    
    html_file = './static/index.html'
    with open(html_file, 'r') as f:
        content = f.read()
    
    # Check header elements
    assert '<header>' in content, "Missing header element"
    assert '<h1>EvidenzLLM</h1>' in content, "Missing or incorrect h1"
    assert 'Evidence-based Question Answering' in content, "Missing subtitle"
    print(f"✓ Header elements present")
    
    # Check chat container
    assert 'id="chat-container"' in content, "Missing chat-container"
    assert 'id="messages"' in content, "Missing messages div"
    print(f"✓ Chat container elements present")
    
    # Check input container
    assert 'id="input-container"' in content, "Missing input-container"
    assert 'id="question-input"' in content, "Missing question input field"
    assert 'id="send-btn"' in content, "Missing send button"
    assert 'placeholder="Ask a question..."' in content, "Missing or incorrect placeholder"
    print(f"✓ Input elements present")
    
    # Check loading indicator
    assert 'id="loading"' in content, "Missing loading div"
    assert 'class="hidden"' in content, "Loading should start hidden"
    print(f"✓ Loading indicator present")


def test_css_structure():
    """Test CSS file structure."""
    print("\nTesting CSS structure...")
    
    css_file = './static/style.css'
    assert os.path.exists(css_file), f"CSS file not found: {css_file}"
    print(f"✓ CSS file exists: {css_file}")
    
    with open(css_file, 'r') as f:
        content = f.read()
    
    # Check for key selectors
    required_selectors = [
        'body',
        '.container',
        'header',
        '#chat-container',
        '#messages',
        '.message',
        '.message.user',
        '.message.assistant',
        '.message.error',
        '.query-type',
        '.evidence',
        '.passage',
        '.passage-header',
        '.passage-text',
        '#input-container',
        '#question-input',
        '#send-btn',
        '#loading',
        '.hidden'
    ]
    
    for selector in required_selectors:
        assert selector in content, f"Missing CSS selector: {selector}"
    
    print(f"✓ All required CSS selectors present ({len(required_selectors)} selectors)")


def test_css_responsive_design():
    """Test responsive design media queries."""
    print("\nTesting responsive design...")
    
    css_file = './static/style.css'
    with open(css_file, 'r') as f:
        content = f.read()
    
    # Check for media queries
    assert '@media' in content, "Missing media queries"
    assert 'max-width: 768px' in content, "Missing tablet breakpoint"
    assert 'max-width: 480px' in content, "Missing mobile breakpoint"
    print(f"✓ Responsive media queries present")
    
    # Check for mobile-specific styles
    media_queries = re.findall(r'@media[^{]+{', content)
    print(f"✓ Found {len(media_queries)} media queries")


def test_css_styling_features():
    """Test CSS styling features."""
    print("\nTesting CSS styling features...")
    
    css_file = './static/style.css'
    with open(css_file, 'r') as f:
        content = f.read()
    
    # Check for gradient
    assert 'linear-gradient' in content, "Missing gradient styling"
    print(f"✓ Gradient styling present")
    
    # Check for animations
    assert 'animation' in content or '@keyframes' in content, "Missing animations"
    print(f"✓ Animations present")
    
    # Check for transitions
    assert 'transition' in content, "Missing transitions"
    print(f"✓ Transitions present")
    
    # Check for box-shadow
    assert 'box-shadow' in content, "Missing box-shadow"
    print(f"✓ Box-shadow styling present")
    
    # Check for border-radius
    assert 'border-radius' in content, "Missing border-radius"
    print(f"✓ Border-radius styling present")


def test_javascript_structure():
    """Test JavaScript file structure."""
    print("\nTesting JavaScript structure...")
    
    js_file = './static/app.js'
    assert os.path.exists(js_file), f"JavaScript file not found: {js_file}"
    print(f"✓ JavaScript file exists: {js_file}")
    
    with open(js_file, 'r') as f:
        content = f.read()
    
    # Check for DOM element references
    assert 'API_URL' in content, "Missing API_URL constant"
    assert 'messagesDiv' in content, "Missing messagesDiv reference"
    assert 'inputField' in content, "Missing inputField reference"
    assert 'sendBtn' in content, "Missing sendBtn reference"
    assert 'loadingDiv' in content, "Missing loadingDiv reference"
    print(f"✓ DOM element references present")
    
    # Check for getElementById calls
    assert "getElementById('messages')" in content, "Missing messages element reference"
    assert "getElementById('question-input')" in content, "Missing input element reference"
    assert "getElementById('send-btn')" in content, "Missing button element reference"
    assert "getElementById('loading')" in content, "Missing loading element reference"
    print(f"✓ All getElementById calls present")


def test_javascript_event_listeners():
    """Test JavaScript event listeners."""
    print("\nTesting JavaScript event listeners...")
    
    js_file = './static/app.js'
    with open(js_file, 'r') as f:
        content = f.read()
    
    # Check for event listeners
    assert 'addEventListener' in content, "Missing event listeners"
    assert "'click'" in content or '"click"' in content, "Missing click event listener"
    assert "'keypress'" in content or '"keypress"' in content, "Missing keypress event listener"
    assert "e.key === 'Enter'" in content or 'e.key === "Enter"' in content, "Missing Enter key handling"
    print(f"✓ Event listeners present (click and keypress)")


def test_javascript_functions():
    """Test JavaScript functions."""
    print("\nTesting JavaScript functions...")
    
    js_file = './static/app.js'
    with open(js_file, 'r') as f:
        content = f.read()
    
    # Check for required functions
    required_functions = [
        'sendQuery',
        'appendMessage',
        'displayAnswer'
    ]
    
    for func in required_functions:
        assert f'function {func}' in content or f'const {func}' in content or f'async function {func}' in content, f"Missing function: {func}"
    
    print(f"✓ All required functions present: {required_functions}")


def test_javascript_api_integration():
    """Test JavaScript API integration."""
    print("\nTesting JavaScript API integration...")
    
    js_file = './static/app.js'
    with open(js_file, 'r') as f:
        content = f.read()
    
    # Check for fetch API usage
    assert 'fetch' in content, "Missing fetch API call"
    assert 'POST' in content, "Missing POST method"
    assert 'Content-Type' in content, "Missing Content-Type header"
    assert 'application/json' in content, "Missing JSON content type"
    assert 'JSON.stringify' in content, "Missing JSON.stringify"
    print(f"✓ Fetch API integration present")
    
    # Check for error handling
    assert 'try' in content and 'catch' in content, "Missing try-catch error handling"
    assert 'finally' in content, "Missing finally block"
    print(f"✓ Error handling present")
    
    # Check for response handling
    assert 'response.ok' in content or 'response.status' in content, "Missing response status check"
    assert 'response.json()' in content, "Missing JSON parsing"
    print(f"✓ Response handling present")


def test_javascript_ui_updates():
    """Test JavaScript UI update logic."""
    print("\nTesting JavaScript UI updates...")
    
    js_file = './static/app.js'
    with open(js_file, 'r') as f:
        content = f.read()
    
    # Check for loading indicator control
    assert 'classList.remove' in content, "Missing classList.remove"
    assert 'classList.add' in content, "Missing classList.add"
    assert "'hidden'" in content or '"hidden"' in content, "Missing hidden class toggle"
    print(f"✓ Loading indicator control present")
    
    # Check for button disable/enable
    assert 'disabled = true' in content or 'disabled=true' in content, "Missing button disable"
    assert 'disabled = false' in content or 'disabled=false' in content, "Missing button enable"
    print(f"✓ Button state control present")
    
    # Check for DOM manipulation
    assert 'createElement' in content, "Missing createElement"
    assert 'appendChild' in content, "Missing appendChild"
    assert 'className' in content or 'classList' in content, "Missing class manipulation"
    assert 'textContent' in content or 'innerHTML' in content, "Missing content manipulation"
    print(f"✓ DOM manipulation methods present")
    
    # Check for scroll behavior
    assert 'scrollTop' in content or 'scrollHeight' in content, "Missing scroll behavior"
    print(f"✓ Scroll behavior present")


def test_javascript_message_display():
    """Test JavaScript message display logic."""
    print("\nTesting JavaScript message display...")
    
    js_file = './static/app.js'
    with open(js_file, 'r') as f:
        content = f.read()
    
    # Check for message types
    assert "'user'" in content or '"user"' in content or 'user' in content, "Missing user message type"
    assert "'assistant'" in content or '"assistant"' in content or 'assistant' in content, "Missing assistant message type"
    assert "'error'" in content or '"error"' in content or 'error' in content, "Missing error message type"
    print(f"✓ Message types present (user, assistant, error)")
    
    # Check for query type display
    assert 'query_type' in content or 'query-type' in content, "Missing query type display"
    print(f"✓ Query type display present")
    
    # Check for evidence display
    assert 'passages' in content, "Missing passages handling"
    assert 'forEach' in content, "Missing forEach for passages"
    assert 'ce_score' in content, "Missing ce_score display"
    assert 'toFixed' in content, "Missing score formatting"
    print(f"✓ Evidence display logic present")
    
    # Check for passage truncation
    assert 'substring' in content or 'slice' in content, "Missing text truncation"
    print(f"✓ Text truncation present")


def test_frontend_integration():
    """Test frontend integration points."""
    print("\nTesting frontend integration...")
    
    # Check that all files are in static directory
    static_dir = './static'
    assert os.path.exists(static_dir), "Static directory not found"
    
    required_files = ['index.html', 'style.css', 'app.js']
    for filename in required_files:
        filepath = os.path.join(static_dir, filename)
        assert os.path.exists(filepath), f"Missing file: {filepath}"
    
    print(f"✓ All frontend files present in static directory")
    
    # Check file sizes
    for filename in required_files:
        filepath = os.path.join(static_dir, filename)
        size = os.path.getsize(filepath)
        print(f"  {filename}: {size} bytes")


def test_accessibility_features():
    """Test accessibility features."""
    print("\nTesting accessibility features...")
    
    html_file = './static/index.html'
    with open(html_file, 'r') as f:
        html_content = f.read()
    
    # Check for lang attribute
    assert 'lang="en"' in html_content, "Missing lang attribute"
    print(f"✓ Language attribute present")
    
    # Check for semantic HTML
    assert '<header>' in html_content, "Missing semantic header"
    print(f"✓ Semantic HTML elements present")
    
    # Check for input attributes
    assert 'placeholder=' in html_content, "Missing placeholder text"
    assert 'autocomplete=' in html_content, "Missing autocomplete attribute"
    print(f"✓ Input accessibility attributes present")


def main():
    """Run all frontend tests."""
    print("=" * 70)
    print("FRONTEND FUNCTIONALITY TESTS")
    print("=" * 70)
    
    try:
        test_html_structure()
        test_html_elements()
        test_css_structure()
        test_css_responsive_design()
        test_css_styling_features()
        test_javascript_structure()
        test_javascript_event_listeners()
        test_javascript_functions()
        test_javascript_api_integration()
        test_javascript_ui_updates()
        test_javascript_message_display()
        test_frontend_integration()
        test_accessibility_features()
        
        print("\n" + "=" * 70)
        print("ALL FRONTEND TESTS PASSED ✓")
        print("=" * 70)
        print("\nNote: To test the frontend in a browser:")
        print("  1. Start the Flask server: python app.py")
        print("  2. Open http://localhost:5000 in your browser")
        print("  3. Test the chat interface manually")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
