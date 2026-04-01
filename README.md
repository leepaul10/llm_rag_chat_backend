# Adaptive RAG Chatbot

A self-triggering Retrieval-Augmented Generation (RAG) chatbot that dynamically decides whether to use vector search or direct LLM inference based on cosine similarity scoring.

## How It Works

1. User sends a message
2. The system encodes the query and searches a FAISS vector index
3. If cosine similarity score > 0.4 → retrieves top 3 relevant chunks and augments the prompt (RAG mode)
4. If score < 0.4 → sends query directly to LLM (Chat mode)
5. Response is generated using Groq's LLaMA 3.3 70B model

## Tech Stack

- **Backend:** Python, FastAPI, FAISS, SentenceTransformers, Groq API
- **Frontend:** HTML, CSS, JavaScript
- **Model:** all-MiniLM-L6-v2 (embeddings), LLaMA 3.3 70B (generation)
- **Scraping:** BeautifulSoup, Requests

## Project Structure

```
rag-chatbot/
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── backend/
│   └── rag/
│       ├── buildindex.py
│       ├── retriever.py
│       ├── index.faiss
│       └── chunks.pkl
├── main.py
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repo
```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
```

2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Add your API key in a `.env` file
```
GROQ_API_KEY=your_key_here
```

5. Build the index
```bash
python backend/rag/buildindex.py
```

6. Run the server
```bash
uvicorn main:app --reload
```

7. Open `frontend/index.html` in your browser 
## Running Locally

## Deployment

- **Backend** is deployed on Render: `https://yourproject.onrender.com`
- **Frontend** is deployed on Vercel: `https://yourproject.vercel.app`

After deployment, update the API URL in `frontend/script.js`:
```javascript
const API_URL = "https://yourproject.onrender.com/chat";
```
## Features

- Self-triggering RAG using cosine similarity threshold
- 200+ Wikipedia articles indexed (28,000+ chunks)
- RAG / AI mode indicator on every response
- FastAPI REST backend with CORS support
