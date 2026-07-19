"""LangGraph node implementations for the corrective RAG workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from app.config import CHROMA_DB_DIR, EMBEDDING_MODEL, GROQ_API_KEY, LLM_MODEL, TOP_K
from app.prompts import (
    ANSWER_GENERATION_SYSTEM_PROMPT,
    DOCUMENT_GRADING_SYSTEM_PROMPT,
    QUERY_ANALYSIS_SYSTEM_PROMPT,
)
from app.schemas import DocumentGradeResult, GenerationResult, QueryAnalysisResult
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


def _build_document_grader_model() -> Any:
    """Create the deterministic Groq model that returns strict JSON grades."""
    model = ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
    )
    return model.with_structured_output(DocumentGradeResult)


def document_grading_node(state: CorrectiveRAGState) -> dict[str, Any]:
    """Strictly grade every retrieved chunk and retain only relevant documents.

    Each model invocation returns a JSON-structured ``relevant`` or ``irrelevant``
    grade. If no chunks pass, this returns an incremented ``retry_count`` so the
    corrective workflow can rewrite and retrieve again.
    """
    question = (state.get("rewritten_question") or state["question"]).strip()
    if not question:
        raise ValueError("A question or rewritten_question is required for document grading")

    retrieved_docs = state.get("retrieved_docs", [])
    if not retrieved_docs:
        return {
            "filtered_docs": [],
            "retry_count": state.get("retry_count", 0) + 1,
        }

    grader = _build_document_grader_model()
    filtered_docs = []
    for document in retrieved_docs:
        response = grader.invoke(
            [
                SystemMessage(content=DOCUMENT_GRADING_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"User question:\n{question}\n\n"
                        f"Retrieved document chunk:\n{document.page_content}"
                    )
                ),
            ]
        )
        grade = (
            response
            if isinstance(response, DocumentGradeResult)
            else DocumentGradeResult.model_validate(response)
        )
        if grade.grade == "relevant":
            filtered_docs.append(document)

    retry_count = state.get("retry_count", 0)
    if not filtered_docs:
        retry_count += 1
    return {"filtered_docs": filtered_docs, "retry_count": retry_count}


def _build_generation_model() -> Any:
    """Create the deterministic Groq model used for grounded answer generation."""
    model = ChatGroq(
        model=LLM_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
    )
    return model.with_structured_output(GenerationResult)


def _document_citation(document: Any, index: int) -> str:
    """Build a stable citation label from the chunk's persisted metadata."""
    metadata = document.metadata
    source = str(metadata.get("filename") or metadata.get("source") or f"document-{index}")
    filename = Path(source).name or f"document-{index}"
    chunk_id = metadata.get("chunk_id")
    return f"{filename} (chunk {chunk_id})" if chunk_id is not None else filename


def generation_node(state: CorrectiveRAGState) -> dict[str, Any]:
    """Generate a cited answer using only relevance-filtered retrieved context."""
    question = (state.get("rewritten_question") or state["question"]).strip()
    if not question:
        raise ValueError("A question or rewritten_question is required for answer generation")

    documents = state.get("filtered_docs", [])
    if not documents:
        return {
            "answer": "The information is unavailable in the retrieved context.",
            "sources": [],
        }

    citations = [_document_citation(document, index) for index, document in enumerate(documents)]
    context = "\n\n".join(
        f"[{citation}]\n{document.page_content}"
        for citation, document in zip(citations, documents)
    )
    response = _build_generation_model().invoke(
        [
            SystemMessage(content=ANSWER_GENERATION_SYSTEM_PROMPT),
            HumanMessage(content=f"Question:\n{question}\n\nContext:\n{context}"),
        ]
    )
    generation = (
        response
        if isinstance(response, GenerationResult)
        else GenerationResult.model_validate(response)
    )
    return {"answer": generation.answer.strip(), "sources": list(dict.fromkeys(citations))}
