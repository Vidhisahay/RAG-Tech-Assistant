from langchain_core.documents import Document

from ingestion.splitter import DocumentSplitter


def test_document_splitter_preserves_metadata_and_assigns_chunk_ids() -> None:
    splitter = DocumentSplitter(chunk_size=20, chunk_overlap=0)
    documents = [
        Document(
            page_content="alpha beta gamma delta epsilon zeta eta theta iota kappa",
            metadata={"source": "docs/guide.md", "filename": "guide.md"},
        ),
        Document(
            page_content="lambda mu nu xi omicron pi rho sigma tau upsilon",
            metadata={"source": "docs/guide.md", "filename": "guide.md"},
        ),
    ]

    chunks = splitter.split(documents)

    assert len(chunks) >= 2
    assert all(chunk.metadata["source"] == "docs/guide.md" for chunk in chunks)
    assert all(chunk.metadata["filename"] == "guide.md" for chunk in chunks)
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == list(range(len(chunks)))