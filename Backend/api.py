"""HTTP API for the document retrieval and question-answering application."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from .ask_documents import (
        DEFAULT_EMBEDDING_MODEL,
        DEFAULT_LLM,
        DEFAULT_RERANKER_MODEL,
        ask_ollama,
        build_prompt,
        get_embedding_model,
        get_reranker_model,
        retrieve_and_rerank,
    )
except ImportError:
    from ask_documents import (
        DEFAULT_EMBEDDING_MODEL,
        DEFAULT_LLM,
        DEFAULT_RERANKER_MODEL,
        ask_ollama,
        build_prompt,
        get_embedding_model,
        get_reranker_model,
        retrieve_and_rerank,
    )

BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "vector_db"
DOCUMENTS_DIR = BASE_DIR / "Documents"
COLLECTION = os.getenv("COLLECTION", "ukg_documents")

app = FastAPI(title="UKG Document Intelligence API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):(\d+)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def preload_retrieval_models() -> None:
    """Warm retrieval models before the first question arrives."""
    get_embedding_model(DEFAULT_EMBEDDING_MODEL)
    get_reranker_model(DEFAULT_RERANKER_MODEL)


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=20)
    candidate_k: int = Field(default=8, ge=1, le=100)


class SearchResponse(BaseModel):
    question: str
    results: list[dict[str, Any]]


class AskResponse(SearchResponse):
    answer: str


def retrieve(request: QueryRequest) -> list[dict[str, Any]]:
    if request.candidate_k < request.top_k:
        raise HTTPException(status_code=400, detail="candidate_k must be >= top_k")
    try:
        return retrieve_and_rerank(
            DB_DIR,
            COLLECTION,
            request.question,
            request.candidate_k,
            request.top_k,
            DEFAULT_EMBEDDING_MODEL,
            DEFAULT_RERANKER_MODEL,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "ukg-document-api"}


@app.get("/api/documents/{filename}")
def document(filename: str) -> FileResponse:
    requested_file = (DOCUMENTS_DIR / Path(filename).name).resolve()
    if requested_file.parent != DOCUMENTS_DIR.resolve() or requested_file.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF documents are available")
    if not requested_file.is_file():
        raise HTTPException(status_code=404, detail="Document not found")
    return FileResponse(requested_file, media_type="application/pdf", filename=requested_file.name)


@app.post("/api/search", response_model=SearchResponse)
def search(request: QueryRequest) -> SearchResponse:
    return SearchResponse(
        question=request.question,
        results=retrieve(request),
    )


@app.post("/api/ask", response_model=AskResponse)
def ask(request: QueryRequest) -> AskResponse:
    results = retrieve(request)
    try:
        answer = ask_ollama(DEFAULT_LLM, build_prompt(request.question, results))
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return AskResponse(question=request.question, answer=answer, results=results)
