"""Compiled LangGraph workflow for corrective retrieval-augmented generation."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.nodes import (
    document_grading_node,
    generation_node,
    query_analysis_node,
    retrieval_node,
)
from app.state import CorrectiveRAGState


def return_unknown_node(_: CorrectiveRAGState) -> dict[str, object]:
    """Provide the terminal response after all corrective retrieval attempts fail."""
    return {"answer": "I don't know.", "sources": []}


def route_after_grading(
    state: CorrectiveRAGState,
) -> Literal["generation", "query_analysis", "return_unknown"]:
    """Choose generation, another corrective attempt, or the terminal fallback."""
    if state.get("filtered_docs"):
        return "generation"
    if state.get("retry_count", 0) < state.get("max_retries", 0):
        return "query_analysis"
    return "return_unknown"


def build_corrective_rag_graph() -> CompiledStateGraph:
    """Build and compile the corrective RAG workflow."""
    workflow = StateGraph(CorrectiveRAGState)
    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("document_grading", document_grading_node)
    workflow.add_node("generation", generation_node)
    workflow.add_node("return_unknown", return_unknown_node)

    workflow.add_edge(START, "query_analysis")
    workflow.add_edge("query_analysis", "retrieval")
    workflow.add_edge("retrieval", "document_grading")
    workflow.add_conditional_edges(
        "document_grading",
        route_after_grading,
        {
            "generation": "generation",
            "query_analysis": "query_analysis",
            "return_unknown": "return_unknown",
        },
    )
    workflow.add_edge("generation", END)
    workflow.add_edge("return_unknown", END)
    return workflow.compile()


corrective_rag_graph = build_corrective_rag_graph()
