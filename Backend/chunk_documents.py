"""Extract and chunk PDF documents into JSONL records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/bert-base-nli-mean-tokens"


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> Iterator[str]:
    """Yield word-based chunks while preserving a configurable overlap."""
    words = re.findall(r"\S+", text)
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            yield chunk
        if start + chunk_size >= len(words):
            break


def iter_pdf_chunks(
    pdf_path: Path,
    chunk_size: int,
    overlap: int,
) -> Iterator[dict[str, object]]:
    """Extract text from each PDF page and yield JSON-serializable chunks."""
    reader = PdfReader(str(pdf_path))

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for chunk_number, text in enumerate(
            split_into_chunks(page_text, chunk_size, overlap), start=1
        ):
            yield {
                "id": f"{pdf_path.stem}-p{page_number}-c{chunk_number}",
                "text": text,
                "metadata": {
                    "source": pdf_path.name,
                    "page": page_number,
                    "chunk": chunk_number,
                },
            }


def find_pdfs(input_path: Path) -> list[Path]:
    """Return one PDF or all PDFs in a directory, in stable order."""
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Input file is not a PDF: {input_path}")
        return [input_path]

    if input_path.is_dir():
        return sorted(input_path.glob("*.pdf"))

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def chunk_documents(
    input_path: Path,
    output_path: Path,
    chunk_size: int,
    overlap: int,
    print_chunks: bool = False,
    embedding_model: str | None = None,
) -> int:
    """Write chunks to JSONL, optionally adding normalized BERT embeddings."""
    if chunk_size <= 0:
        raise ValueError("chunk-size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least zero and smaller than chunk-size")

    pdfs = find_pdfs(input_path)
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in: {input_path}")

    records = [
        record
        for pdf_path in pdfs
        for record in iter_pdf_chunks(pdf_path, chunk_size, overlap)
    ]

    if embedding_model:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Embedding requires sentence-transformers. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from error

        model = SentenceTransformer(embedding_model)
        embeddings = model.encode(
            [record["text"] for record in records],
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        for record, embedding in zip(records, embeddings):
            record["embedding"] = embedding.tolist()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            serialized_record = json.dumps(record, ensure_ascii=False)
            output_file.write(serialized_record + "\n")
            if print_chunks:
                print(serialized_record)

    return len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text from PDF documents and write overlapping JSONL chunks."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="A PDF file or a directory containing PDF files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("chunks.jsonl"),
        help="Output JSONL path (default: chunks.jsonl).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Maximum words per chunk (default: 800).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=120,
        help="Words repeated between adjacent chunks (default: 120).",
    )
    parser.add_argument(
        "--print-chunks",
        action="store_true",
        help="Print every chunk to the terminal as JSONL.",
    )
    parser.add_argument(
        "--embed-model",
        default=None,
        help=(
            "Add normalized embeddings using this sentence-transformers model "
            f"(default BERT model: {DEFAULT_EMBEDDING_MODEL})."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = chunk_documents(
        args.input,
        args.output,
        args.chunk_size,
        args.overlap,
        args.print_chunks,
        args.embed_model,
    )
    print(f"Wrote {count} chunks to {args.output}")


if __name__ == "__main__":
    main()
