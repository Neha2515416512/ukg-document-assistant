"""Retrieve, rerank, and answer document questions with a local Ollama model."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/bert-base-nli-mean-tokens"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_LLM = "qwen2.5:7b"
DEFAULT_COLLECTION = "ukg_documents"
ANSI_ESCAPE = re.compile(r"\x1B(?:[@-_][0-?]*[ -/]*[@-~])")


def retrieve_and_rerank(
    db_dir: Path,
    collection_name: str,
    question: str,
    candidate_k: int,
    top_k: int,
    embedding_model_name: str,
    reranker_model_name: str,
) -> list[dict[str, Any]]:
    """Retrieve vector candidates and return the best cross-encoder matches."""
    if top_k <= 0 or candidate_k < top_k:
        raise ValueError("candidate-k must be greater than or equal to top-k, and top-k must be positive")

    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_collection(collection_name)
    collection_size = collection.count()
    if collection_size == 0:
        raise ValueError("The vector collection is empty. Ingest embeddings first.")

    embedding_model = SentenceTransformer(embedding_model_name)
    query_embedding = embedding_model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()
    search_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(candidate_k, collection_size),
        include=["documents", "metadatas", "distances"],
    )

    candidates = [
        {
            "text": text,
            "metadata": metadata,
            "vector_distance": distance,
        }
        for text, metadata, distance in zip(
            search_results["documents"][0],
            search_results["metadatas"][0],
            search_results["distances"][0],
        )
    ]
    reranker = CrossEncoder(reranker_model_name)
    scores = reranker.predict([(question, item["text"]) for item in candidates])
    for item, score in zip(candidates, scores):
        item["rerank_score"] = float(score)

    return sorted(
        candidates,
        key=lambda item: item["rerank_score"],
        reverse=True,
    )[:top_k]


def build_prompt(question: str, results: list[dict[str, Any]]) -> str:
    """Build a grounded prompt with source metadata for the local LLM."""
    context = "\n\n".join(
        f"Source: {result['metadata'].get('source', 'unknown')}, "
        f"page {result['metadata'].get('page', 'unknown')}\n"
        f"{result['text']}"
        for result in results
    )
    return (
        "You are a document question-answering assistant. "
        "Answer only from the context below. "
        "If the answer is not in the context, say you could not find it. "
        "Give a concise answer and cite the source filename and page number.\n\n"
        f"Question: {question}\n\nContext:\n{context}\n"
    )


def ask_ollama(llm: str, prompt: str) -> str:
    """Send the grounded prompt to Ollama and return its answer."""
    completed = subprocess.run(
        ["ollama", "run", llm, prompt],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "Ollama returned an error")
    return ANSI_ESCAPE.sub("", completed.stdout).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask questions against the local document vector database."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask about the documents.",
    )
    parser.add_argument("--db-dir", type=Path, default=Path("vector_db"))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--llm", default=DEFAULT_LLM)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print the reranked source chunks before the LLM answer.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Ask multiple questions in one terminal session.",
    )
    return parser.parse_args()


def answer_question(args: argparse.Namespace, question: str) -> None:
    results = retrieve_and_rerank(
        args.db_dir,
        args.collection,
        question,
        args.candidate_k,
        args.top_k,
        args.embedding_model,
        args.reranker_model,
    )

    if args.show_sources:
        print("Reranked sources:")
        for rank, result in enumerate(results, start=1):
            print(json.dumps({"rank": rank, **result}, ensure_ascii=False, indent=2))
        print("\nLLM answer:")

    print(ask_ollama(args.llm, build_prompt(question, results)))


def main() -> None:
    args = parse_args()
    if args.interactive or not args.question:
        print("Document Q&A ready. Type a question, or type 'exit' to stop.")
        while True:
            try:
                question = input("\nQuestion: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break
            if question.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break
            if not question:
                continue
            try:
                answer_question(args, question)
            except Exception as error:
                print(f"Error: {error}")
    else:
        answer_question(args, args.question)


if __name__ == "__main__":
    main()
