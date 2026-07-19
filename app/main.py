"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api import router


app = FastAPI(
    title="RAG Tech Assistant",
    version="1.0.0",
    description="Corrective RAG-based Technical Documentation Assistant built with LangGraph"
)

app.include_router(router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "RAG Technical Documentation Assistant API",
        "status": "running",
        "docs": "/docs",
        "openapi": "/openapi.json"
    }
