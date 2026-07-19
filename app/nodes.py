"""LangGraph node implementations for the corrective RAG workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.config import CHROMA_DB_DIR, EMBEDDING_MODEL, GROQ_API_KEY, LLM_MODEL, TOP_K
from app.prompts import QUERY_ANALYSIS_SYSTEM_PROMPT
from app.schemas import QueryAnalysisResult
from app.state import CorrectiveRAGState
from ingestion.embeddings import SentenceTransformerEmbeddings
from ingestion.vectorstore import ChromaVectorStore


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


def _load_retrieval_store() -> ChromaVectorStore | None:
    """Open the ingested Chroma collection, if one has been persisted locally."""
    if not Path(CHROMA_DB_DIR).exists():
        return None
    return ChromaVectorStore.load_persisted(
        persist_directory=CHROMA_DB_DIR,
        embeddings=SentenceTransformerEmbeddings(EMBEDDING_MODEL),
    )


def retrieval_node(state: CorrectiveRAGState) -> dict[str, list[Any]]:
    """Retrieve the five most relevant persisted chunks as LangChain Documents.

    The returned mapping is a LangGraph state update. Missing, un-ingested, and empty
    Chroma collections all produce an empty ``retrieved_docs`` list.
    """
    query = (state.get("rewritten_question") or state["question"]).strip()
    if not query:
        raise ValueError("A question or rewritten_question is required for retrieval")

    store = _load_retrieval_store()
    if store is None or store.count == 0:
        return {"retrieved_docs": []}
    return {"retrieved_docs": store.similarity_search(query, limit=TOP_K)}
