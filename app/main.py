"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api import router


app = FastAPI(title="RAG Tech Assistant")
app.include_router(router)
