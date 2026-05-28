"""
FastAPI service for IntelliDesk triage.

Health, readiness, info, single-ticket triage, and batch triage endpoints.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import CATEGORIES, PRIORITIES
from src.pipeline import TicketTriagePipeline

logger = logging.getLogger("intellidesk.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


class TriageRequest(BaseModel):
    """Single triage request."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Raw ticket description (preprocessing applied internally)",
        examples=["VPN keeps disconnecting from corporate network"],
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Number of KB suggestions to return (1-10)",
    )


class BatchTriageRequest(BaseModel):
    """Batch triage request, up to 100 tickets per call."""
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of raw ticket descriptions",
    )
    top_k: Optional[int] = Field(default=None, ge=1, le=10)


class KBSuggestion(BaseModel):
    """One retrieved KB article."""
    kb_id: str
    title: str
    category: str
    content: str
    tags: List[str]
    score: float


class TriageResponse(BaseModel):
    """Single triage result."""
    ticket_text: str
    category: str
    category_confidence: float
    priority: str
    priority_confidence: float
    kb_suggestions: List[KBSuggestion]
    needs_human_review: bool
    review_reasons: List[str]
    latency_ms: float


class BatchTriageResponse(BaseModel):
    results: List[TriageResponse]
    total_count: int


class HealthResponse(BaseModel):
    status: str
    service: str


class ReadyResponse(BaseModel):
    ready: bool
    pipeline_loaded: bool


class InfoResponse(BaseModel):
    service: str
    version: str
    categories: List[str]
    priorities: List[str]
    kb_articles: int
    config: dict


pipeline: Optional[TicketTriagePipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context: warms up pipeline at startup, logs shutdown."""
    global pipeline
    logger.info("Starting IntelliDesk API — warming up pipeline...")
    pipeline = TicketTriagePipeline()
    pipeline.warm_up()
    logger.info("Pipeline ready. API serving requests.")
    yield
    logger.info("Shutting down IntelliDesk API")


SERVICE_NAME = "intellidesk-api"
SERVICE_VERSION = "0.1.0"

app = FastAPI(
    title="IntelliDesk Triage API",
    description=(
        "AI-powered IT service desk ticket triage. "
        "Classifies tickets by category and priority, and recommends "
        "the most relevant KB resolution articles via semantic search."
    ),
    version=SERVICE_VERSION,
    lifespan=lifespan,
)

# Open CORS by default for local dev. In production, replace allow_origins
# with the exact frontend domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_ready_pipeline() -> TicketTriagePipeline:
    """Raise 503 if pipeline isn't loaded (shouldn't happen post-lifespan)."""
    if pipeline is None or not pipeline.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not loaded — check service logs",
        )
    return pipeline


@app.get("/", include_in_schema=False)
async def root():
    """Root: points users at the interactive docs."""
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": ["/health", "/ready", "/info", "/triage", "/triage/batch"],
    }


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health():
    """
    Liveness probe: always returns 200 if the process is up.
    Use for k8s livenessProbe / Docker HEALTHCHECK.
    """
    return HealthResponse(status="ok", service=SERVICE_NAME)


@app.get("/ready", response_model=ReadyResponse, tags=["health"])
async def ready():
    """
    Readiness probe: 200 only when pipeline is loaded and serving.
    Use for k8s readinessProbe to gate traffic to this pod.
    """
    is_ready = pipeline is not None and pipeline.is_ready
    return ReadyResponse(ready=is_ready, pipeline_loaded=is_ready)


@app.get("/info", response_model=InfoResponse, tags=["service"])
async def info():
    """Service metadata: model labels, KB size, config thresholds."""
    p = _require_ready_pipeline()
    config = {
        "top_k": p.config.top_k,
        "category_confidence_threshold": p.config.category_confidence_threshold,
        "priority_confidence_threshold": p.config.priority_confidence_threshold,
        "min_kb_score": p.config.min_kb_score,
    }
    kb_count = p._retriever.n_articles if p._retriever else 0
    return InfoResponse(
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        categories=list(CATEGORIES),
        priorities=list(PRIORITIES),
        kb_articles=kb_count,
        config=config,
    )


@app.post("/triage", response_model=TriageResponse, tags=["triage"])
async def triage(request: TriageRequest):
    """
    Triage a single ticket: classify category and priority, return KB suggestions.

    Pass raw ticket text; preprocessing is handled internally. Response
    includes confidence scores and a human-review flag with explanations
    of why review was triggered.
    """
    p = _require_ready_pipeline()
    try:
        result = p.triage(request.text, top_k=request.top_k)
        return TriageResponse(**result.to_dict())
    except Exception as exc:
        logger.exception("Triage failed for ticket")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Triage failed: {exc}",
        )


@app.post("/triage/batch", response_model=BatchTriageResponse, tags=["triage"])
async def triage_batch(request: BatchTriageRequest):
    """Process up to 100 tickets in one call."""
    p = _require_ready_pipeline()
    try:
        results = p.triage_batch(request.texts, top_k=request.top_k)
        return BatchTriageResponse(
            results=[TriageResponse(**r.to_dict()) for r in results],
            total_count=len(results),
        )
    except Exception as exc:
        logger.exception("Batch triage failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch triage failed: {exc}",
        )


def main() -> None:
    """Run uvicorn programmatically. Honors env vars for host/port/reload."""
    import uvicorn

    host = os.getenv("INTELLIDESK_HOST", "0.0.0.0")
    port = int(os.getenv("INTELLIDESK_PORT", "8000"))
    reload = os.getenv("INTELLIDESK_RELOAD", "false").lower() == "true"

    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
