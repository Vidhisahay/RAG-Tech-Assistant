"""FastAPI route handlers for querying, ingestion, and user feedback."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from langchain_core.documents import Document

from app.config import CHROMA_DB_DIR, EMBEDDING_MODEL, MAX_RETRIES
from app.graph import corrective_rag_graph
from app.schemas import (
    DocumentsResponse,
    FeedbackRequest,
    FeedbackResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from ingestion.embeddings import SentenceTransformerEmbeddings
from ingestion.ingest import IngestionConfig, ingest_documents, ingest_urls
from ingestion.vectorstore import ChromaVectorStore


router = APIRouter()
FEEDBACK_FILE = Path("feedback.json")


def _validate_markdown_upload(filename: str | None) -> str:
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded files must include a filename.",
        )
    if Path(filename).suffix.lower() != ".md":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only Markdown files are supported: {filename}",
        )
    return filename


def _load_feedback_entries() -> list[dict[str, object]]:
    if not FEEDBACK_FILE.exists():
        return []
    with FEEDBACK_FILE.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else []


def _persist_feedback_entry(payload: FeedbackRequest) -> None:
    entries = _load_feedback_entries()
    entries.append(
        {
            "thumbs_up": payload.thumbs_up,
            "comment": payload.comment,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    with FEEDBACK_FILE.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)


@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
def query(request: QueryRequest) -> QueryResponse:
    """Run the compiled corrective RAG graph for one user question."""
    result = corrective_rag_graph.invoke(
        {
            "question": request.question.strip(),
            "retry_count": 0,
            "max_retries": MAX_RETRIES,
        }
    )
    return QueryResponse(
        answer=str(result.get("answer", "")).strip(),
        sources=[str(source) for source in result.get("sources", [])],
    )


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_200_OK)
async def ingest(
    files: list[UploadFile] | None = File(default=None),
    urls: list[str] | None = Form(default=None),
) -> IngestResponse:
    """Ingest uploaded Markdown files and/or explicit documentation URLs."""
    config = IngestionConfig()
    uploaded_docs: list[Document] = []

    if files:
        for upload in files:
            filename = _validate_markdown_upload(upload.filename)
            content = (await upload.read()).decode("utf-8", errors="replace")
            uploaded_docs.append(
                Document(
                    page_content=content,
                    metadata={"filename": filename, "source": f"upload:{filename}"},
                )
            )

    valid_urls = [url.strip() for url in (urls or []) if url.strip()]
    if not uploaded_docs and not valid_urls:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide at least one Markdown file or URL for ingestion.",
        )

    vectors_stored = 0
    if uploaded_docs:
        vectors_stored += ingest_documents(uploaded_docs, config)
    if valid_urls:
        vectors_stored += ingest_urls(valid_urls, config)

    return IngestResponse(
        files_ingested=len(uploaded_docs),
        urls_ingested=len(valid_urls),
        vectors_stored=vectors_stored,
    )


@router.get("/documents", response_model=DocumentsResponse, status_code=status.HTTP_200_OK)
def list_documents() -> DocumentsResponse:
    """Return unique indexed document names from the persisted vector store."""
    store = ChromaVectorStore.load_persisted(
        persist_directory=CHROMA_DB_DIR,
        embeddings=SentenceTransformerEmbeddings(EMBEDDING_MODEL),
    )
    if store is None:
        return DocumentsResponse(document_names=[])
    return DocumentsResponse(document_names=store.list_document_names())


@router.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Persist user feedback to a local feedback.json file."""
    _persist_feedback_entry(request)
    return FeedbackResponse(message="Feedback saved.")
