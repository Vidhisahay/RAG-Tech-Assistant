from app.nodes import retrieval_node

state = {
    "question": "How do I define a request body in FastAPI?",
    "retrieved_docs": [],
    "filtered_docs": [],
    "retry_count": 0,
}

result = retrieval_node(state)

docs = result["retrieved_docs"]

print(f"Retrieved {len(docs)} documents\n")

for i, doc in enumerate(docs):
    print("=" * 50)
    print(f"Document {i+1}")
    print(doc.metadata)
    print(doc.page_content[:300])