"""Embedding adapter backed by Sentence Transformers."""

from __future__ import annotations

from typing import Any, Sequence


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEmbeddings:
    """Expose one embedding model through document and query-oriented methods."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str | None = None) -> None:
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Sentence Transformers and a compatible PyTorch installation are required for embeddings."
            ) from error
        self._model: Any = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def __call__(self, input: Sequence[str]) -> list[list[float]]:
        """Compatibility hook for Chroma embedding-function consumers."""
        return self.embed_documents(input)
