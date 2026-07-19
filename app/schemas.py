"""Structured LLM responses used by workflow nodes."""

from typing import Literal

from pydantic import BaseModel, Field


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
