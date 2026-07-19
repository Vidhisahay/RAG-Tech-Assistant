"""Streamlit frontend for the RAG Technical Documentation Assistant.

This UI only consumes the FastAPI API and does not implement LangGraph logic.
"""

from __future__ import annotations

from typing import Any

import requests
import streamlit as st


DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"
QUERY_ENDPOINT = "/query"
REQUEST_TIMEOUT_SECONDS = 60


def ask_backend(backend_url: str, question: str) -> dict[str, Any]:
    """Send one question to the FastAPI backend and return parsed JSON."""
    response = requests.post(
        f"{backend_url.rstrip('/')}{QUERY_ENDPOINT}",
        json={"question": question},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def render_sources(sources: list[str]) -> None:
    """Render source citations returned by the backend."""
    st.subheader("Source Citations")
    if not sources:
        st.info("No source citations were returned.")
        return
    for source in sources:
        st.markdown(f"- {source}")


def main() -> None:
    st.set_page_config(page_title="RAG Technical Documentation Assistant", page_icon="📚")

    st.title("RAG Technical Documentation Assistant")
    st.caption("Ask technical documentation questions using the FastAPI backend.")

    question = st.text_area(
        "Question",
        placeholder="Ask a question about your technical documentation...",
        height=160,
    )

    if st.button("Ask", type="primary"):
        trimmed_question = question.strip()
        if not trimmed_question:
            st.warning("Please enter a question before submitting.")
            return

        try:
            with st.spinner("Thinking..."):
                result = ask_backend(DEFAULT_BACKEND_URL, trimmed_question)
        except requests.exceptions.ConnectionError:
            st.error(
                "Unable to connect to the backend. Make sure FastAPI is running at the configured URL."
            )
            return
        except requests.exceptions.Timeout:
            st.error("The backend request timed out. Please try again.")
            return
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            st.error(f"Backend returned an HTTP error (status: {status_code}).")
            return
        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")
            return

        answer = str(result.get("answer", "")).strip()
        sources = result.get("sources", [])

        st.subheader("Answer")
        st.write(answer or "No answer was returned.")
        render_sources([str(source) for source in sources])


if __name__ == "__main__":
    main()