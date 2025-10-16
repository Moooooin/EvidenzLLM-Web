# EvidenzLLM Web Chat - Setup & Troubleshooting Guide

Complete setup instructions and solutions to common issues.

## 📋 Table of Contents

1. [Detailed Installation](#detailed-installation)
2. [Data Preparation](#data-preparation)
3. [Configuration](#configuration)
4. [Running the Application](#running-the-application)
5. [Troubleshooting](#troubleshooting)
6. [Development](#development)

---

## 📦 Detailed Installation

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Disk Space**: ~2GB for models and data
- **OS**: macOS, Linux, or Windows

### Step-by-Step Installation

#### 1. Clone or Download the Project

```bash
cd /path/to/project
```

#### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**
- `torch` - PyTorch for ML models
- `transformers` - Hugging Face transformers
- `sentence-transformers` - Dense embeddings
- `faiss-cpu` - Vector similarity search
- `rank-bm25` - Sparse retrieval
- `google-generativeai` - Gemini API
- `flask` - Web server
- `flask-cors` - CORS support
- `nltk` - Text processing
- `python-dotenv` - Environment variables

#### 4. Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt')"
```

This downloads the sentence tokenizer needed for text chunking.

---

## 📚 Data Preparation

### Query Classifier Model

The query classifier model should be in `query_classifier_model/` with these files:

- `model.safetensors` or `pytorch_model.bin` (model weights)
- `vocab.json`, `merges.txt` (tokenizer vocabulary)
- `tokenizer.json`, `tokenizer_config.json` (tokenizer config)
- `special_tokens_map.json` (special tokens)

**Note**: These files come from the original EvidenzLLM notebook training process.

### Wikipedia Knowledge Base

Generate the Wikipedia data:

```bash
python prepare_data.py
```

**What it does:**
1. Fetches Wikipedia articles on configured topics
2. Chunks text into overlapping segments
3. Saves processed data to `data/wiki_texts.pkl`

**Customize topics** by editing `prepare_data.py`:

```python
topics = [
    "Machine Learning",
    "Artificial Intelligence",
    "Physics",
    "Theory of Relativity",
    "Your Custom Topic"
]
```

**Expected output:**
- File: `data/wiki_texts.pkl`
- Size: Several MB
- Time: 2-5 minutes

---

## ⚙️ Configuration

### Environment Variables

Create `.env` file:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Required: Your Google Gemini API key
GOOGLE_API_KEY=your-actual-api-key-here

# Optional: Model and data paths
CLASSIFIER_PATH=./query_classifier_model
WIKI_DATA_PATH=./data/wiki_texts.pkl

# Optional: Gemini model selection
GEMINI_MODEL=gemini-2.0-flash

# Optional: Server configuration
PORT=5000
```

### Getting a Gemini API Key

1. Go to https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key to your `.env` file

### Model Selection

Available Gemini models:
- `gemini-2.0-flash` (recommended) - Fast, stable, good safety filter support
- `gemini-1.5-flash` - Older but reliable
- `gemini-1.5-pro` - More capable but slower
- `gemini-2.0-flash-exp` - Experimental features

---

## 🚀 Running the Application

### Method 1: Direct Start (Recommended)

```bash
python app.py
```

### Method 2: Using Startup Script

```bash
./start_server.sh
```

The script automatically sets environment variables for macOS compatibility.

### Method 3: Development Mode

```bash
export FLASK_ENV=development
python app.py
```

### Expected Startup

```
======================================================================
EvidenzLLM Web Chat - Starting Server
======================================================================

Using device: cpu
Loading tokenizer...
Loading query classifier model...
Loading Wikipedia data...
Building chunks...
Initializing hybrid retriever...
Found 10 unique topics
Initializing Gemini generator...
Pipeline initialization complete!

======================================================================
Server starting on http://localhost:5000
Press Ctrl+C to stop
======================================================================
```

**First startup**: 1-2 minutes (model loading)  
**Subsequent starts**: 30-60 seconds

---

## 🔧 Troubleshooting

### 1. Threading/Mutex Errors on macOS

**Symptom:**
```
mutex lock failed: Invalid argument
zsh: abort      python app.py
```

**Solution:**
```bash
export TOKENIZERS_PARALLELISM=false
export MKL_THREADING_LAYER=GNU
python app.py
```

Or use `./start_server.sh` which handles this automatically.

### 2. Model Loading Errors

**Symptom:**
```
FileNotFoundError: No model file found
```

**Solution:**
1. Verify model directory exists:
   ```bash
   ls -la query_classifier_model/
   ```

2. Check for required files:
   - `model.safetensors` OR `pytorch_model.bin`
   - Tokenizer files

3. Verify path in `.env`:
   ```bash
   CLASSIFIER_PATH=./query_classifier_model
   ```

### 3. Wikipedia Data Missing

**Symptom:**
```
FileNotFoundError: './data/wiki_texts.pkl'
```

**Solution:**
```bash
python prepare_data.py
```

Verify the file was created:
```bash
ls -lh data/wiki_texts.pkl
```

### 4. API Key Problems

**Symptom:**
```
ValueError: GOOGLE_API_KEY environment variable is required
```

**Solution:**
1. Check `.env` file exists and contains your key
2. Verify no extra spaces or quotes around the key
3. Test the key at https://makersuite.google.com/app/apikey

**Symptom:**
```
RuntimeError: Gemini API error: 400 API key not valid
```

**Solution:**
1. Generate a new API key
2. Ensure Gemini API is enabled in Google Cloud Console
3. Check API quota hasn't been exceeded

### 5. Safety Filter Blocking

**Symptom:**
```
Response blocked by safety filters
```

**Solution:**
The app uses `BLOCK_NONE` safety settings, but some content may still be blocked. Try:
1. Rephrasing your question
2. Using a different Gemini model (e.g., `gemini-1.5-flash`)
3. Checking if the content is genuinely problematic

### 6. Port Already in Use

**Symptom:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Change port in .env
PORT=5001

# Or kill process using port 5000
kill -9 $(lsof -ti:5000)
```

### 7. Import Errors

**Symptom:**
```
ModuleNotFoundError: No module named 'flask'
```

**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

Verify installation:
```bash
pip list | grep -E "flask|torch|transformers"
```

### 8. NLTK Data Missing

**Symptom:**
```
LookupError: Resource punkt not found
```

**Solution:**
```bash
python -c "import nltk; nltk.download('punkt')"
```

### 9. Memory Issues

**Symptom:**
```
Killed: 9
```
or slow performance

**Solution:**
1. Close other applications
2. Use CPU instead of GPU (automatic fallback)
3. Reduce retrieval candidates in `retrieval/retrieval.py`
4. Consider upgrading RAM

### 10. Slow Performance

**Solutions:**
1. Use GPU if available
2. Reduce `top_k` in retrieval (default: 5)
3. Use smaller embedding models
4. Cache frequent queries

---

## 👨‍💻 Development

### Running Tests

```bash
# Run all tests
./tests/run_all_tests.sh

# Run specific test
python tests/test_pipeline.py
```

### Debug Mode

```bash
export FLASK_ENV=development
python app.py
```

### Logging

Redirect logs to file:
```bash
python app.py > app.log 2>&1
```

### Code Structure

- `app.py` - Flask server and API endpoints
- `pipeline/pipeline.py` - Main orchestrator
- `models/models.py` - Query classifier
- `retrieval/retrieval.py` - Hybrid retrieval
- `generator/generator.py` - Gemini integration
- `static/` - Frontend files

### Adding New Wikipedia Topics

Edit `prepare_data.py`:

```python
topics = [
    "Your New Topic",
    "Another Topic"
]
```

Then regenerate data:
```bash
python prepare_data.py
```

---

## 📊 Performance Optimization

### GPU Acceleration

If you have CUDA-capable GPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Reduce Memory Usage

Edit `retrieval/retrieval.py`:

```python
def retrieve(self, query, top_k=5, alpha=0.6,
             bm25_candidates=100,  # reduced from 200
             dense_candidates=100,  # reduced from 200
             rerank_top=16):  # reduced from 32
```

### Cache Results

Implement caching for repeated queries (not included by default).

---

## 🔒 Security Best Practices

1. **Never commit `.env` file** - Contains API key
2. **Use `.gitignore`** - Excludes sensitive files
3. **Rotate API keys** - Regularly update keys
4. **Monitor API usage** - Check Google Cloud Console
5. **Limit API access** - Use API key restrictions

---

## 📞 Getting Help

If you're still experiencing issues:

1. Check the console logs for detailed error messages
2. Verify your environment matches requirements
3. Try the test suite to identify failing components
4. Search for similar issues
5. Create a detailed bug report with:
   - Error message
   - Steps to reproduce
   - Environment details (OS, Python version)
   - Relevant logs

---

## 🔍 Verification Checklist

Before running, verify:

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip list`)
- [ ] NLTK data downloaded
- [ ] `.env` file created with API key
- [ ] Model files in `query_classifier_model/`
- [ ] Wikipedia data generated (`data/wiki_texts.pkl`)
- [ ] Port 5000 available (or configured differently)

---

## 📈 Monitoring

### Check API Usage

Monitor your Gemini API usage at:
https://console.cloud.google.com/apis/dashboard

### Performance Metrics

- **Response time**: 2-4 seconds per query
- **Memory usage**: ~4GB RAM
- **API calls**: 1 per query
- **Token usage**: ~500-1000 tokens per query

---

Last Updated: 2025-10-16
