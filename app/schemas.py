"""Structured LLM responses used by workflow nodes."""

from typing import Literal

from pydantic import AnyHttpUrl, BaseModel, Field


class QueryAnalysisResult(BaseModel):
    """Validated result of rewriting and classifying a user question."""

    rewritten_question: str = Field(
        description="A retrieval-ready question that preserves the user's original intent."
    )
    query_type: Literal["conceptual", "how_to", "troubleshooting", "api_reference"] = Field(
        description="The question category selected from the supported corrective-RAG categories."
    )


class DocumentGradeResult(BaseModel):
    """Validated strict relevance grade for one retrieved document chunk."""

    grade: Literal["relevant", "irrelevant"] = Field(
        description="Whether the chunk directly helps answer the user's question."
    )


class GenerationResult(BaseModel):
    """Validated answer generated exclusively from the provided RAG context."""

    answer: str = Field(
        description="A grounded answer with metadata-derived citations, or the required unavailable response."
    )


class IngestUrlRequest(BaseModel):
    """Request body for explicit, on-demand URL ingestion."""

    url: AnyHttpUrl = Field(description="The documentation URL to load through WebBaseLoader.")


class QueryRequest(BaseModel):
    """Incoming user query to run through the corrective RAG graph."""

    question: str = Field(min_length=1, description="The user question to answer.")


class QueryResponse(BaseModel):
    """Answer and source citations returned by the corrective RAG graph."""

    answer: str
    sources: list[str]


class IngestResponse(BaseModel):
    """Summary of an explicit ingestion run."""

    files_ingested: int
    urls_ingested: int
    vectors_stored: int


class DocumentsResponse(BaseModel):
    """Indexed unique document filenames."""

    document_names: list[str]


class FeedbackRequest(BaseModel):
    """Simple user feedback payload."""

    thumbs_up: bool
    comment: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    """Confirmation response for persisted feedback."""

    message: str
