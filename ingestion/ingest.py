"""Command-line entry point and orchestration for document ingestion."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ingestion.embeddings import DEFAULT_EMBEDDING_MODEL, SentenceTransformerEmbeddings
from ingestion.loader import CompositeDocumentLoader, LocalDocumentLoader, WebDocumentLoader
from ingestion.splitter import DocumentSplitter
from ingestion.vectorstore import ChromaVectorStore


@dataclass(frozen=True)
class IngestionConfig:
    docs_directory: Path = Path("docs")
    persist_directory: Path = Path("chroma_db")
    urls: tuple[str, ...] = field(default_factory=tuple)
    chunk_size: int = 700
    chunk_overlap: int = 100
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    collection_name: str = "documentation"


class IngestionPipeline:
    """Coordinate loading, splitting, embedding, and persistence."""

    def __init__(self, config: IngestionConfig) -> None:
        self._config = config

    def run(self) -> int:
        loader = CompositeDocumentLoader(
            [LocalDocumentLoader(self._config.docs_directory), WebDocumentLoader(self._config.urls)]
        )
        chunks = DocumentSplitter(self._config.chunk_size, self._config.chunk_overlap).split(loader.load())
        if not chunks:
            return 0
        store = ChromaVectorStore(
            self._config.persist_directory,
            SentenceTransformerEmbeddings(self._config.embedding_model),
            self._config.collection_name,
        )
        return store.upsert(chunks)


def _parse_args(args: Sequence[str] | None = None) -> IngestionConfig:
    parser = argparse.ArgumentParser(description="Ingest local and web documentation into ChromaDB.")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--persist-dir", default="chroma_db")
    parser.add_argument("--url", action="append", default=[], help="Documentation URL; repeat as needed.")
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--collection", default="documentation")
    parsed = parser.parse_args(args)
    return IngestionConfig(
        docs_directory=Path(parsed.docs_dir),
        persist_directory=Path(parsed.persist_dir),
        urls=tuple(parsed.url),
        chunk_size=parsed.chunk_size,
        chunk_overlap=parsed.chunk_overlap,
        collection_name=parsed.collection,
    )


def main(args: Sequence[str] | None = None) -> None:
    count = IngestionPipeline(_parse_args(args)).run()
    print(f"Persisted {count} document chunks to ChromaDB.")


if __name__ == "__main__":
    main()
