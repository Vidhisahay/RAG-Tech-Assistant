from pathlib import Path

from langchain_core.documents import Document

from ingestion.loader import LocalDocumentLoader, WebDocumentLoader


def test_local_document_loader_loads_only_markdown_files(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    nested_dir = docs_dir / "nested"
    nested_dir.mkdir(parents=True)
    (docs_dir / "intro.md").write_text("# Intro", encoding="utf-8")
    (nested_dir / "guide.md").write_text("# Guide", encoding="utf-8")
    (docs_dir / "ignore.txt").write_text("ignore me", encoding="utf-8")

    documents = LocalDocumentLoader(docs_dir).load()

    assert [doc.metadata["filename"] for doc in documents] == ["intro.md", "guide.md"]
    assert [Path(doc.metadata["source"]).name for doc in documents] == ["intro.md", "guide.md"]
    assert all(doc.page_content.startswith("#") for doc in documents)


def test_web_document_loader_filters_placeholder_urls(monkeypatch) -> None:
    captured_paths: list[str] = []

    class FakeWebBaseLoader:
        def __init__(self, web_paths):
            captured_paths.extend(web_paths)

        def load(self):
            return [Document(page_content="web content", metadata={})]

    monkeypatch.setattr("ingestion.loader.WebBaseLoader", FakeWebBaseLoader)

    documents = WebDocumentLoader(
        ["https://example.com/docs", "https://docs.example.org/guide.md"]
    ).load()

    assert captured_paths == ["https://docs.example.org/guide.md"]
    assert documents[0].metadata["source"] == "https://docs.example.org/guide.md"
    assert documents[0].metadata["filename"] == "guide.md"