"""Store embedded document chunks in a persistent Chroma vector database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chromadb

DEFAULT_COLLECTION = "ukg_documents"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/bert-base-nli-mean-tokens"
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_records(input_path: Path) -> list[dict[str, Any]]:
    """Load JSONL chunk records that contain text, metadata, and embeddings."""
    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}") from error

            if not record.get("id") or not record.get("text"):
                raise ValueError(f"Record on line {line_number} needs id and text")
            if not record.get("embedding"):
                raise ValueError(
                    f"Record on line {line_number} has no embedding. "
                    "Run chunk_documents.py with --embed-model first."
                )
            records.append(record)

    if not records:
        raise ValueError(f"No records found in {input_path}")
    return records


def get_collection(persist_dir: Path, collection_name: str):
    """Open or create a cosine-similarity Chroma collection."""
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def add_records(collection: Any, records: list[dict[str, Any]]) -> None:
    """Upsert document chunks and their embeddings into Chroma."""
    collection.upsert(
        ids=[record["id"] for record in records],
        documents=[record["text"] for record in records],
        metadatas=[record.get("metadata", {}) for record in records],
        embeddings=[record["embedding"] for record in records],
    )


def query_collection(
    persist_dir: Path,
    collection_name: str,
    query_text: str,
    top_k: int,
    model_name: str,
    candidate_k: int,
    reranker_model_name: str,
) -> None:
    """Retrieve candidates with Chroma, rerank them, and print the best matches."""
    if top_k <= 0:
        raise ValueError("top-k must be greater than zero")
    if candidate_k < top_k:
        raise ValueError("candidate-k must be greater than or equal to top-k")

    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Querying requires sentence-transformers. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from error

    collection = get_collection(persist_dir, collection_name)
    if collection.count() == 0:
        raise ValueError("The collection is empty. Ingest embeddings first.")

    model = SentenceTransformer(model_name)
    query_embedding = model.encode(
        query_text,
        normalize_embeddings=True,
    ).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(candidate_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    candidates = list(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ))
    reranker = CrossEncoder(reranker_model_name)
    rerank_scores = reranker.predict(
        [(query_text, document) for document, _, _ in candidates]
    )
    ranked_candidates = sorted(
        zip(candidates, rerank_scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )[:top_k]

    for rank, ((document, metadata, distance), rerank_score) in enumerate(
        ranked_candidates,
        start=1,
    ):
        print(json.dumps({
            "rank": rank,
            "rerank_score": float(rerank_score),
            "vector_distance": distance,
            "metadata": metadata,
            "text": document,
        }, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest embedded chunks into Chroma and query them semantically."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("chunks_with_embeddings.jsonl"),
        help="Embedded JSONL file (default: chunks_with_embeddings.jsonl).",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=Path("vector_db"),
        help="Persistent Chroma database directory (default: vector_db).",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Chroma collection name (default: {DEFAULT_COLLECTION}).",
    )
    parser.add_argument(
        "--query",
        help="Search text. If omitted, the embedded chunks are ingested only.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5).",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Number of vector-search candidates to rerank (default: 20).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Model used to embed queries (default: {DEFAULT_EMBEDDING_MODEL}).",
    )
    parser.add_argument(
        "--reranker-model",
        default=DEFAULT_RERANKER_MODEL,
        help=(
            "Cross-encoder model used to rerank candidates "
            f"(default: {DEFAULT_RERANKER_MODEL})."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Embedding file not found: {args.input}")

    collection = get_collection(args.db_dir, args.collection)
    records = load_records(args.input)
    add_records(collection, records)
    print(f"Stored {len(records)} chunks in '{args.db_dir}/{args.collection}'")

    if args.query:
        query_collection(
            args.db_dir,
            args.collection,
            args.query,
            args.top_k,
            args.model,
            args.candidate_k,
            args.reranker_model,
        )


if __name__ == "__main__":
    main()
