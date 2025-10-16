// DOM element references
const API_URL = '/api/query';
const TOPICS_URL = '/api/topics';
const messagesDiv = document.getElementById('messages');
const inputField = document.getElementById('question-input');
const sendBtn = document.getElementById('send-btn');
const loadingDiv = document.getElementById('loading');

// Event listeners
sendBtn.addEventListener('click', sendQuery);
inputField.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendQuery();
});

// Load available topics on page load
window.addEventListener('DOMContentLoaded', loadAvailableTopics);

// Send query function
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
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Request failed');
        }

        const data = await response.json();
        console.log('Received data:', data);
        displayAnswer(data);

    } catch (error) {
        appendMessage('error', `Error: ${error.message}`);
    } finally {
        loadingDiv.classList.add('hidden');
        sendBtn.disabled = false;
    }
}

// Helper function to append simple messages
function appendMessage(type, content) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${type}`;
    msgDiv.textContent = content;
    messagesDiv.appendChild(msgDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Display answer with evidence
function displayAnswer(data) {
    // Validate data structure
    if (!data || typeof data !== 'object') {
        appendMessage('error', 'Error: Invalid response format');
        console.error('Invalid data:', data);
        return;
    }

    if (!data.answer) {
        appendMessage('error', 'Error: No answer in response');
        console.error('Missing answer in data:', data);
        return;
    }

    // Answer message
    const answerDiv = document.createElement('div');
    answerDiv.className = 'message assistant';

    const queryTypeSpan = document.createElement('span');
    queryTypeSpan.className = 'query-type';
    queryTypeSpan.textContent = `[${data.query_type || 'unknown'}]`;

    const answerText = document.createElement('p');
    answerText.textContent = data.answer;

    answerDiv.appendChild(queryTypeSpan);
    answerDiv.appendChild(answerText);

    // Evidence passages
    if (data.passages && Array.isArray(data.passages) && data.passages.length > 0) {
        const evidenceDiv = document.createElement('div');
        evidenceDiv.className = 'evidence';
        evidenceDiv.innerHTML = '<strong>Evidence:</strong>';

        data.passages.forEach((passage, idx) => {
            const passageDiv = document.createElement('div');
            passageDiv.className = 'passage';
            passageDiv.innerHTML = `
                <div class="passage-header">
                    [${idx + 1}] ${passage.title || 'Unknown'}
                </div>
                <div class="passage-text">${(passage.chunk || '').substring(0, 200)}...</div>
            `;
            evidenceDiv.appendChild(passageDiv);
        });

        answerDiv.appendChild(evidenceDiv);
    }

    messagesDiv.appendChild(answerDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Load available topics from database
async function loadAvailableTopics() {
    try {
        const response = await fetch(TOPICS_URL);
        if (!response.ok) {
            console.error('Failed to load topics');
            return;
        }
        
        const data = await response.json();
        if (data.topics && Array.isArray(data.topics)) {
            displayTopics(data.topics);
        }
    } catch (error) {
        console.error('Error loading topics:', error);
    }
}

// Display topics in header
function displayTopics(topics) {
    const topicsContainer = document.getElementById('wiki-topics');
    if (!topicsContainer) return;

    // Clear existing topics
    topicsContainer.innerHTML = '';

    // Add topic tags (limit to first 10 for display)
    const displayTopics = topics.slice(0, 10);
    displayTopics.forEach(topic => {
        const topicTag = document.createElement('span');
        topicTag.className = 'wiki-topic-tag';
        topicTag.textContent = topic;
        topicTag.title = topic; // Show full name on hover
        topicsContainer.appendChild(topicTag);
    });
    
    // Add "and more" indicator if there are more topics
    if (topics.length > 10) {
        const moreTag = document.createElement('span');
        moreTag.className = 'wiki-topic-tag wiki-topic-more';
        moreTag.textContent = `+${topics.length - 10} more`;
        moreTag.title = `Total: ${topics.length} topics`;
        topicsContainer.appendChild(moreTag);
    }
}
