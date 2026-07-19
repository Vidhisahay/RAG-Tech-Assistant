"""Command-line entry point and orchestration for document ingestion."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document

from ingestion.embeddings import DEFAULT_EMBEDDING_MODEL, SentenceTransformerEmbeddings
from ingestion.loader import DocumentLoader, LocalDocumentLoader, WebDocumentLoader
from ingestion.splitter import DocumentSplitter
from ingestion.vectorstore import ChromaVectorStore


@dataclass(frozen=True)
class IngestionConfig:
    docs_directory: Path = Path("docs")
    persist_directory: Path = Path("chroma_db")
    chunk_size: int = 700
    chunk_overlap: int = 100
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    collection_name: str = "documentation"


@dataclass(frozen=True)
class IngestionStats:
    files_loaded: int
    chunks_created: int
    vectors_stored: int


class IngestionPipeline:
    """Build the default corpus from local Markdown files only."""

    def __init__(self, config: IngestionConfig) -> None:
        self._config = config

    def run(self) -> IngestionStats:
        return _ingest(LocalDocumentLoader(self._config.docs_directory), self._config)


def ingest_urls(urls: Sequence[str], config: IngestionConfig) -> int:
    """Ingest URL documentation on demand; never used by the default corpus run."""
    return _ingest(WebDocumentLoader(urls), config).vectors_stored


def ingest_documents(documents: Sequence[Document], config: IngestionConfig) -> int:
    """Ingest in-memory documents, used by explicit upload endpoints."""

    class InMemoryDocumentLoader:
        def __init__(self, docs: Sequence[Document]) -> None:
            self._docs = list(docs)

        def load(self) -> list[Document]:
            return list(self._docs)

    return _ingest(InMemoryDocumentLoader(documents), config).vectors_stored


def _ingest(loader: DocumentLoader, config: IngestionConfig) -> IngestionStats:
    """Split, embed, and persist documents emitted by one explicit source adapter."""
    documents = loader.load()
    chunks = DocumentSplitter(config.chunk_size, config.chunk_overlap).split(documents)
    if not chunks:
        return IngestionStats(
            files_loaded=len(documents),
            chunks_created=0,
            vectors_stored=0,
        )
    store = ChromaVectorStore(
        config.persist_directory,
        SentenceTransformerEmbeddings(config.embedding_model),
        config.collection_name,
    )
    vectors_stored = store.upsert(chunks)
    return IngestionStats(
        files_loaded=len(documents),
        chunks_created=len(chunks),
        vectors_stored=vectors_stored,
    )


def _parse_args(args: Sequence[str] | None = None) -> IngestionConfig:
    parser = argparse.ArgumentParser(description="Ingest local Markdown documentation into ChromaDB.")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--persist-dir", default="chroma_db")
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--collection", default="documentation")
    parsed = parser.parse_args(args)
    return IngestionConfig(
        docs_directory=Path(parsed.docs_dir),
        persist_directory=Path(parsed.persist_dir),
        chunk_size=parsed.chunk_size,
        chunk_overlap=parsed.chunk_overlap,
        collection_name=parsed.collection,
    )


def main(args: Sequence[str] | None = None) -> None:
    stats = IngestionPipeline(_parse_args(args)).run()
    print(f"Files loaded: {stats.files_loaded}")
    print(f"Chunks created: {stats.chunks_created}")
    print(f"Vectors stored: {stats.vectors_stored}")
    print("Indexing completed successfully.")


if __name__ == "__main__":
    main()
