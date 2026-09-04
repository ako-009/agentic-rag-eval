from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Agentic Self-Healing RAG API",
    description="RAG pipeline with critic agent, self-healing loop, and LLM eval CI/CD",
    version="1.0.0",
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "service": "Agentic Self-Healing RAG",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "POST /api/v1/query",
            "POST /api/v1/ingest",
            "POST /api/v1/eval",
            "GET  /api/v1/health",
        ]
    }