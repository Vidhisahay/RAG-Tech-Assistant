"""Document source adapters for local files and web pages."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol, Sequence
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".htm"}


class DocumentLoader(Protocol):
    """Contract implemented by every source of documents."""

    def load(self) -> list[Document]:
        """Return documents enriched with source and filename metadata."""


class LocalDocumentLoader:
    """Load Markdown, text, and HTML files from a directory recursively."""

    def __init__(self, docs_directory: str | Path) -> None:
        self._docs_directory = Path(docs_directory)

    def load(self) -> list[Document]:
        if not self._docs_directory.exists():
            raise FileNotFoundError(f"Documents directory does not exist: {self._docs_directory}")
        if not self._docs_directory.is_dir():
            raise NotADirectoryError(f"Documents path is not a directory: {self._docs_directory}")

        documents: list[Document] = []
        for path in sorted(self._docs_directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            documents.append(self._load_file(path))
        return documents

    @staticmethod
    def _load_file(path: Path) -> Document:
        content = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix.lower() in {".html", ".htm"}:
            content = BeautifulSoup(content, "html.parser").get_text("\n", strip=True)

        return Document(
            page_content=content,
            metadata={"source": str(path.resolve()), "filename": path.name},
        )


class WebDocumentLoader:
    """Load web documentation through LangChain's :class:`WebBaseLoader`."""

    def __init__(self, urls: Sequence[str]) -> None:
        self._urls = tuple(url for url in urls if url.strip())

    def load(self) -> list[Document]:
        if not self._urls:
            return []

        documents = WebBaseLoader(web_paths=self._urls).load()
        for index, document in enumerate(documents):
            source = str(document.metadata.get("source") or self._urls[index % len(self._urls)])
            document.metadata.update({"source": source, "filename": self._filename(source)})
        return documents

    @staticmethod
    def _filename(url: str) -> str:
        parsed = urlparse(url)
        name = Path(parsed.path).name
        return name or parsed.netloc or url


class CompositeDocumentLoader:
    """Combine independent loaders without coupling the pipeline to source types."""

    def __init__(self, loaders: Iterable[DocumentLoader]) -> None:
        self._loaders = tuple(loaders)

    def load(self) -> list[Document]:
        return [document for loader in self._loaders for document in loader.load()]
