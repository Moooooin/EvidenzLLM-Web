# EvidenzLLM Web Chat

A modern web-based question answering system that uses evidence-based retrieval and Google Gemini to provide accurate, well-supported answers with a clean ChatGPT-like interface.

## ✨ Features

- **🎯 Smart Query Classification**: Automatically categorizes questions (factual, explanation, reasoning, calculation)
- **🔍 Hybrid Retrieval**: Combines BM25, dense embeddings, and cross-encoder reranking
- **📚 Evidence-Based Answers**: Generates answers using Google Gemini with retrieved Wikipedia evidence
- **💬 Modern Chat UI**: Clean, responsive interface with light effects and smooth animations
- **🏷️ Topic Tags**: Displays available Wikipedia topics in the header
- **⚡ Fast & Reliable**: Optimized pipeline with proper error handling

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# 3. Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# 4. Generate Wikipedia knowledge base
python prepare_data.py
```

### Download Pre-Trained Classifier
You need the query classifier model from the original EvidenzLLM notebook. Place it as `query_classifier_model/` in this directory.

Get a pretrained classifier [here](https://drive.google.com/drive/folders/11eLqNQHzvGy6KBFfqlZJigcynLETLNdH?usp=sharing)

### Run

```bash
# Start the server
python app.py

# Or use the startup script (handles environment variables)
./start_server.sh
```

Open http://localhost:5000 in your browser and start asking questions!

## 📁 Project Structure

```
.
├── app.py                      # Flask backend server
├── pipeline/
│   └── pipeline.py            # Main pipeline orchestrator
├── models/
│   └── models.py              # Query classifier model
├── retrieval/
│   └── retrieval.py           # Hybrid retrieval system
├── generator/
│   └── generator.py           # Gemini API integration
├── static/
│   ├── index.html             # Chat interface
│   ├── style.css              # Modern styling
│   └── app.js                 # Frontend logic
├── query_classifier_model/    # Pre-trained classifier (not in repo)
├── data/
│   └── wiki_texts.pkl         # Wikipedia knowledge base (not in repo)
├── requirements.txt           # Python dependencies
├── .env.example               # Example configuration
└── README.md                  # This file
```

## ⚙️ Configuration

Edit `.env` to customize:

```bash
# Required
GOOGLE_API_KEY=your-api-key-here

# Optional (with defaults)
CLASSIFIER_PATH=./query_classifier_model
WIKI_DATA_PATH=./data/wiki_texts.pkl
GEMINI_MODEL=gemini-2.0-flash
PORT=5000
```

## 🎨 Interface

- **Topic Tags**: Shows available Wikipedia topics in rounded gray pills
- **Query Type Badges**: Displays the classified query type
- **Evidence Passages**: Shows supporting evidence with dark gray borders

## 🔌 API Endpoints

### GET /api/health
Health check endpoint

### GET /api/topics
Get available Wikipedia topics

### POST /api/query
Process a question

**Request:**
```json
{
  "question": "What is machine learning?"
}
```

**Response:**
```json
{
  "question": "What is machine learning?",
  "query_type": "explanation",
  "answer": "Machine learning is...",
  "passages": [...]
}
```

## 🧪 Example Questions

- **Factual**: "Who discovered gravity?"
- **Explanation**: "What is machine learning?"
- **Reasoning**: "Why does the sky appear blue?"
- **Calculation**: "Calculate 15% of 200"

## 🛠️ Troubleshooting

### macOS Threading Error
```bash
export TOKENIZERS_PARALLELISM=false
export MKL_THREADING_LAYER=GNU
python app.py
```

Or just use `./start_server.sh` which handles this automatically.


### API Key Issues
- Verify your key in `.env` file
- Check it's active at https://makersuite.google.com/app/apikey
- Ensure Gemini API is enabled

For more troubleshooting, see [SETUP.md](SETUP.md).

## 📊 Performance

- **First request**: 3-5 seconds (model loading)
- **Subsequent requests**: 2-4 seconds
- **Memory**: ~4GB RAM recommended
- **GPU**: Automatically used if available

## 🔒 Security Notes

- Never commit `.env` file (contains API key)
- Model files and data are excluded from git
- API key is required but kept secure

## 📝 License

This project is based on the EvidenzLLM notebook and uses various open-source models and libraries.

## 🙏 Acknowledgments

- Query Classifier: Fine-tuned DeBERTa model
- Dense Retrieval: sentence-transformers multi-qa-mpnet-base-dot-v1
- Cross-Encoder: ms-marco-MiniLM-L-6-v2
- Answer Generation: Google Gemini
- Knowledge Base: Wikipedia

---

**Need help?** Check [SETUP.md](SETUP.md) for detailed setup instructions and troubleshooting.
