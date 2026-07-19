"""Document chunking service."""

from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentSplitter:
    """Split documents while retaining traceable metadata on every chunk."""

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 100) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Return chunks with source, filename, and per-source ``chunk_id``."""
        chunks = self._splitter.split_documents(documents)
        chunk_counts: defaultdict[str, int] = defaultdict(int)

        for chunk in chunks:
            source = str(chunk.metadata.get("source", "unknown"))
            chunk.metadata.setdefault("filename", "unknown")
            chunk.metadata["chunk_id"] = chunk_counts[source]
            chunk_counts[source] += 1
        return chunks
