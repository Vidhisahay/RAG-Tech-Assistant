from types import SimpleNamespace

from langchain_core.documents import Document

from app import nodes
from app.schemas import DocumentGradeResult, GenerationResult, QueryAnalysisResult


def test_retrieval_node_uses_rewritten_question_and_top_k(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(nodes, "CHROMA_DB_DIR", str(tmp_path))
    fake_docs = [Document(page_content="answer", metadata={"source": "docs/a.md"})]

    class FakeStore:
        count = 1

        def similarity_search(self, query: str, limit: int):
            assert query == "rewritten question"
            assert limit == nodes.TOP_K
            return fake_docs

    monkeypatch.setattr(nodes.ChromaVectorStore, "load_persisted", lambda **kwargs: FakeStore())

    result = nodes.retrieval_node({"question": "original question", "rewritten_question": "rewritten question"})

    assert result == {"retrieved_docs": fake_docs}


def test_query_analysis_node_uses_mocked_llm(monkeypatch) -> None:
    class FakeModel:
        def invoke(self, messages):
            assert messages[1].content == "How do I ingest docs?"
            return {
                "rewritten_question": "How do I ingest documentation?",
                "query_type": "how_to",
            }

    monkeypatch.setattr(nodes, "_build_query_analysis_model", lambda: FakeModel())

    result = nodes.query_analysis_node({"question": "How do I ingest docs?"})

    assert result == {
        "rewritten_question": "How do I ingest documentation?",
        "query_type": "how_to",
    }


def test_document_grading_node_filters_relevant_documents(monkeypatch) -> None:
    docs = [
        Document(page_content="relevant chunk", metadata={"source": "docs/a.md"}),
        Document(page_content="irrelevant chunk", metadata={"source": "docs/b.md"}),
    ]

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return {"grade": "relevant"}
            return DocumentGradeResult(grade="irrelevant")

    monkeypatch.setattr(nodes, "_build_document_grader_model", lambda: FakeModel())

    result = nodes.document_grading_node(
        {"question": "What is this?", "retrieved_docs": docs, "retry_count": 0}
    )

    assert result["filtered_docs"] == [docs[0]]
    assert result["retry_count"] == 0


def test_document_grading_node_increments_retry_when_no_docs() -> None:
    result = nodes.document_grading_node({"question": "What is this?", "retrieved_docs": [], "retry_count": 1})

    assert result == {"filtered_docs": [], "retry_count": 2}


def test_generation_node_uses_mocked_llm_and_returns_sources(monkeypatch) -> None:
    docs = [
        Document(
            page_content="First chunk",
            metadata={"source": "docs/a.md", "filename": "a.md", "chunk_id": 0},
        ),
        Document(
            page_content="Second chunk",
            metadata={"source": "docs/b.md", "filename": "b.md", "chunk_id": 2},
        ),
    ]

    class FakeModel:
        def invoke(self, messages):
            assert "Question:" in messages[1].content
            assert "Context:" in messages[1].content
            return {"answer": " Grounded answer. "}

    monkeypatch.setattr(nodes, "_build_generation_model", lambda: FakeModel())

    result = nodes.generation_node({"question": "Explain the docs", "filtered_docs": docs})

    assert result == {
        "answer": "Grounded answer.",
        "sources": ["a.md (chunk 0)", "b.md (chunk 2)"],
    }