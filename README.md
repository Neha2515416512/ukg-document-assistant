# UKG Document Assistant

A local document question-answering application for UKG knowledge sources. It combines PDF ingestion, semantic retrieval, cross-encoder reranking, and Ollama-powered answer generation behind a FastAPI service with a React interface.

## Highlights

- Extracts and overlaps text chunks from PDF documents
- Generates BERT-family embeddings with Sentence Transformers
- Stores and searches vectors with ChromaDB
- Reranks candidate passages before answer generation
- Returns grounded answers with source passages
- Provides a lightweight React and Vite frontend
- Runs locally without sending documents to a hosted service

## Architecture

```text
PDF documents
    -> chunk_documents.py
    -> sentence-transformer embeddings
    -> ChromaDB vector store
    -> semantic candidate retrieval
    -> cross-encoder reranking
    -> Ollama answer generation
    -> FastAPI API
    -> React frontend
```

## Project Structure

```text
Backend/
  api.py                 FastAPI application
  ask_documents.py      Retrieval, reranking, prompts, and Ollama calls
  chunk_documents.py    PDF parsing and overlapping chunk creation
  vector_database.py    ChromaDB loading and querying
  requirements.txt      Python dependencies
Frontend/
  src/App.jsx           React application and API client
  src/App.css           Application styling
  package.json          Frontend scripts and dependencies
.gitignore
```

## Requirements

- Python 3.11 or newer
- Node.js 18 or newer
- Ollama with the configured local language model
- The source PDFs placed in `Backend/Documents/`

## Setup

### Backend

```powershell
cd Backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Build the chunk and vector data for your documents using the scripts in `Backend`, then start the API:

```powershell
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

The API is available at `http://127.0.0.1:8000`. Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

The backend defaults to the locally installed `orca-mini:latest` model for faster responses. Set `LLM=qwen2.5:7b` before starting the API if you prefer the larger model.

### Frontend

```powershell
cd Frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5174
```

Open `http://127.0.0.1:5174` in a browser.

## API Endpoints

### Health check

```http
GET /api/health
```

### Search documents

```http
POST /api/search
Content-Type: application/json

{
  "question": "How do I correct a missed punch?",
  "top_k": 5,
  "candidate_k": 20
}
```

### Ask a question

```http
POST /api/ask
Content-Type: application/json

{
  "question": "How do I correct a missed punch?"
}
```

## Development Checks

```powershell
cd Frontend
npm run lint
npm run build
```

## Notes

Local PDFs, generated chunks, embeddings, ChromaDB files, virtual environments, and build output are excluded from version control. See `.gitignore` for the complete list.
