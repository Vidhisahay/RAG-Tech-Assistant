"""Shared state schema for the corrective RAG LangGraph workflow."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document


class CorrectiveRAGState(TypedDict):
    """Values passed between retrieval, correction, and answer-generation nodes."""

    question: Annotated[str, "The user's original question, retained unchanged for traceability."]
    rewritten_question: Annotated[
        str,
        "The improved retrieval query produced when the original question needs clarification or correction.",
    ]
    query_type: Annotated[
        Literal["conceptual", "how_to", "troubleshooting", "api_reference"],
        "The question category selected by query analysis for downstream retrieval and answer behavior.",
    ]
    retrieved_docs: Annotated[
        list[Document],
        "All documents returned by the selected retriever before relevance evaluation.",
    ]
    filtered_docs: Annotated[
        list[Document],
        "Retrieved documents judged relevant enough to use as answer context.",
    ]
    answer: Annotated[str, "The final grounded answer generated for the user."]
    sources: Annotated[
        list[str],
        "Source identifiers or URLs cited by the final answer.",
    ]
    retry_count: Annotated[
        int,
        "How many corrective retrieval attempts have already been made for this request.",
    ]
    max_retries: Annotated[
        int,
        "Maximum corrective retrieval attempts permitted before the workflow stops retrying.",
    ]
