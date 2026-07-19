"""LangGraph node implementations for the corrective RAG workflow."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.config import GROQ_API_KEY, LLM_MODEL
from app.prompts import QUERY_ANALYSIS_SYSTEM_PROMPT
from app.schemas import QueryAnalysisResult
from app.state import CorrectiveRAGState


def _build_query_analysis_model() -> Any:
    """Create the deterministic Groq model used to analyze retrieval queries."""
    model = ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
    )
    return model.with_structured_output(QueryAnalysisResult)


def query_analysis_node(state: CorrectiveRAGState) -> dict[str, Any]:
    """Rewrite and classify ``question``, returning the LangGraph state update.

    LangGraph merges this partial mapping into ``CorrectiveRAGState``; all unrelated
    state fields are deliberately preserved by the graph runtime.
    """
    question = state["question"].strip()
    if not question:
        raise ValueError("question must not be empty")

    response = _build_query_analysis_model().invoke(
        [
            SystemMessage(content=QUERY_ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]
    )
    analysis = (
        response
        if isinstance(response, QueryAnalysisResult)
        else QueryAnalysisResult.model_validate(response)
    )
    return {
        "rewritten_question": analysis.rewritten_question.strip(),
        "query_type": analysis.query_type,
    }
