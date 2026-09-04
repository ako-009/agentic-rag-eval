from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import time

from app.graph.builder import rag_graph
from app.retrieval.vectorstore import ingest_documents
from app.eval.pipeline import run_eval_pipeline

router = APIRouter()


# ── Request/Response Models ──────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    k: int = 3  # number of chunks to retrieve


class QueryResponse(BaseModel):
    question: str
    answer: str
    faithfulness_score: float
    is_hallucinated: bool
    retry_count: int
    latency_ms: float


class IngestRequest(BaseModel):
    documents: list[str]
    source: Optional[str] = "api_upload"


class IngestResponse(BaseModel):
    message: str
    documents_ingested: int


class EvalResponse(BaseModel):
    approved: bool
    hallucination_rate: float
    avg_faithfulness: float
    total_evaluated: int
    failures: list[str]


# ── Endpoints ────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main RAG query endpoint.
    Runs the full self-healing pipeline and returns a grounded answer.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    start = time.time()

    initial_state = {
        "query": request.question,
        "reformulated_query": None,
        "retrieved_chunks": [],
        "answer": "",
        "faithfulness_score": 0.0,
        "is_hallucinated": False,
        "critic_reason": "",
        "retry_count": 0,
        "final_answer": "",
    }

    try:
        result = rag_graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {str(e)}")

    latency_ms = (time.time() - start) * 1000

    return QueryResponse(
        question=request.question,
        answer=result["final_answer"],
        faithfulness_score=result["faithfulness_score"],
        is_hallucinated=result["is_hallucinated"],
        retry_count=result["retry_count"],
        latency_ms=latency_ms,
    )


@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(request: IngestRequest):
    """
    Document ingestion endpoint.
    Adds new documents to the ChromaDB knowledge base.
    """
    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    metadatas = [{"source": request.source}] * len(request.documents)

    try:
        ingest_documents(request.documents, metadatas)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")

    return IngestResponse(
        message="Documents ingested successfully",
        documents_ingested=len(request.documents),
    )


@router.post("/eval", response_model=EvalResponse)
async def eval_endpoint():
    """
    Trigger the eval CI/CD pipeline.
    Returns deployment gate result and key metrics.
    """
    try:
        result = run_eval_pipeline(max_questions=5, save_results=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eval error: {str(e)}")

    metrics = result["metrics"]
    gate = result["gate_result"]

    return EvalResponse(
        approved=gate["approved"],
        hallucination_rate=metrics.get("hallucination_rate", 0),
        avg_faithfulness=metrics.get("avg_faithfulness", 0),
        total_evaluated=metrics.get("total_evaluated", 0),
        failures=gate.get("failures", []),
    )


@router.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "agentic-rag-eval"}