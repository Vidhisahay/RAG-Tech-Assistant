"""Persistent ChromaDB storage adapter."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Sequence

import chromadb
from chromadb.api.models.Collection import Collection
from langchain_core.documents import Document

from ingestion.embeddings import SentenceTransformerEmbeddings


class ChromaVectorStore:
    """Store and query document chunks in a local, persistent Chroma collection."""

    def __init__(
        self,
        persist_directory: str | Path,
        embeddings: SentenceTransformerEmbeddings,
        collection_name: str = "documentation",
    ) -> None:
        self._client = chromadb.PersistentClient(path=str(Path(persist_directory)))
        self._collection: Collection = self._client.get_or_create_collection(name=collection_name)
        self._embeddings = embeddings

    def upsert(self, documents: Sequence[Document]) -> int:
        """Embed and persist chunks. Re-ingesting identical chunks is idempotent."""
        if not documents:
            return 0

        texts = [document.page_content for document in documents]
        self._collection.upsert(
            ids=[self._document_id(document) for document in documents],
            documents=texts,
            embeddings=self._embeddings.embed_documents(texts),
            metadatas=[dict(document.metadata) for document in documents],
        )
        return len(documents)

    def similarity_search(self, query: str, limit: int = 5) -> list[Document]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        result = self._collection.query(
            query_embeddings=[self._embeddings.embed_query(query)],
            n_results=limit,
            include=["documents", "metadatas"],
        )
        return [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(result["documents"][0], result["metadatas"][0])
        ]

    @property
    def count(self) -> int:
        return self._collection.count()

    @staticmethod
    def _document_id(document: Document) -> str:
        source = str(document.metadata.get("source", ""))
        chunk_id = str(document.metadata.get("chunk_id", ""))
        value = f"{source}\0{chunk_id}\0{document.page_content}"
        return sha256(value.encode("utf-8")).hexdigest()
