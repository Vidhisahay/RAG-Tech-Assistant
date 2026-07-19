import json
from pathlib import Path

from fastapi.testclient import TestClient

from app import api
from app.main import app


client = TestClient(app)


def test_post_query_returns_answer_and_sources(monkeypatch) -> None:
    class FakeGraph:
        def invoke(self, payload):
            assert payload["question"] == "What is RAG?"
            assert payload["retry_count"] == 0
            return {"answer": "RAG answer", "sources": ["docs/a.md#chunk-0"]}

    monkeypatch.setattr(api, "corrective_rag_graph", FakeGraph())

    response = client.post("/query", json={"question": "What is RAG?"})

    assert response.status_code == 200
    assert response.json() == {"answer": "RAG answer", "sources": ["docs/a.md#chunk-0"]}


def test_post_ingest_accepts_files(monkeypatch) -> None:
    monkeypatch.setattr(api, "ingest_documents", lambda documents, config: len(documents) * 10)
    monkeypatch.setattr(api, "ingest_urls", lambda urls, config: len(urls) * 20)

    response = client.post(
        "/ingest",
        files=[
            ("files", ("one.md", b"# One", "text/markdown")),
            ("files", ("two.md", b"# Two", "text/markdown")),
        ],
    )

    assert response.status_code == 200
    assert response.json() == {"files_ingested": 2, "urls_ingested": 0, "vectors_stored": 20}


def test_post_ingest_accepts_urls(monkeypatch) -> None:
    monkeypatch.setattr(api, "ingest_documents", lambda documents, config: len(documents) * 10)
    monkeypatch.setattr(api, "ingest_urls", lambda urls, config: len(urls) * 20)

    response = client.post(
        "/ingest",
        files=[
            ("urls", (None, "https://docs.example.org/one")),
            ("urls", (None, "https://docs.example.org/two")),
        ],
    )

    assert response.status_code == 200
    assert response.json() == {"files_ingested": 0, "urls_ingested": 2, "vectors_stored": 40}


def test_get_documents_returns_indexed_names(monkeypatch, tmp_path) -> None:
    class FakeStore:
        def list_document_names(self):
            return ["a.md", "b.md"]

    monkeypatch.setattr(api, "SentenceTransformerEmbeddings", lambda model: object())
    monkeypatch.setattr(api.ChromaVectorStore, "load_persisted", lambda **kwargs: FakeStore())

    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == {"document_names": ["a.md", "b.md"]}


def test_post_feedback_persists_json(monkeypatch, tmp_path) -> None:
    feedback_file = tmp_path / "feedback.json"
    monkeypatch.setattr(api, "FEEDBACK_FILE", feedback_file)

    response = client.post("/feedback", json={"thumbs_up": True, "comment": "Helpful"})

    assert response.status_code == 201
    assert response.json() == {"message": "Feedback saved."}

    saved = json.loads(feedback_file.read_text(encoding="utf-8"))
    assert len(saved) == 1
    assert saved[0]["thumbs_up"] is True
    assert saved[0]["comment"] == "Helpful"
    assert "created_at" in saved[0]