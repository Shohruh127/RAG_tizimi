"""
FastAPI application entry point for the Time-Aware Hybrid RAG System.

Endpoints:
  POST /api/v1/ingest   – Ingest a document (metadata + chunks).
  POST /api/v1/search   – Hybrid search with time-travel + LLM answer generation.
  GET  /health          – Health-check.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from database import close_all, get_db_session, init_all, es_client, qdrant_client
from ingestion import ingest_document
from models import DocumentInput, IngestionResponse, SearchRequest, SearchResponse
from retrieval import generate_answer, hybrid_retrieve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

USE_LLM_EXTRACTOR: bool = os.getenv("USE_LLM_EXTRACTOR", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise storage backends on startup; close connections on shutdown."""
    await init_all()
    yield
    await close_all()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Time-Aware Hybrid RAG API",
    description=(
        "Legal Document Search and RAG API with SCD Type 2 versioning, "
        "hybrid dense+sparse retrieval, and time-travel point-in-time queries."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
async def health_check() -> dict:
    """Simple liveness probe."""
    return {"status": "ok"}


@app.post(
    "/api/v1/ingest",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["ingestion"],
    summary="Ingest a legal document with its chunks",
)
async def ingest_endpoint(
    payload: DocumentInput,
    session: AsyncSession = Depends(get_db_session),
) -> IngestionResponse:
    """
    Ingest a document and all its versioned chunks into PostgreSQL,
    Elasticsearch, and Qdrant.
    """
    try:
        doc_orm = await ingest_document(session, payload)
    except Exception as exc:
        logger.exception("Ingestion failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion error: {exc}",
        )

    return IngestionResponse(
        doc_id=str(doc_orm.doc_id),
        chunks_ingested=len(payload.chunks),
        message="Document ingested successfully.",
    )


@app.post(
    "/api/v1/search",
    response_model=SearchResponse,
    tags=["search"],
    summary="Hybrid semantic + keyword search with LLM-generated answer",
)
async def search_endpoint(request: SearchRequest) -> SearchResponse:
    """
    Perform a hybrid (dense + sparse) search over legal chunks with optional
    time-travel filtering, then generate an LLM answer with citations.
    """
    try:
        chunks, pit = await hybrid_retrieve(
            qdrant=qdrant_client,
            es=es_client,
            query=request.query,
            top_k=request.top_k,
            use_llm_extractor=USE_LLM_EXTRACTOR,
        )
    except Exception as exc:
        logger.exception("Retrieval failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval error: {exc}",
        )

    try:
        answer = await generate_answer(request.query, chunks, pit)
    except Exception as exc:
        logger.exception("Generation failed: %s", exc)
        answer = "Answer generation failed."

    return SearchResponse(
        query=request.query,
        point_in_time=pit,
        answer=answer,
        chunks=chunks,
    )


# ---------------------------------------------------------------------------
# Dev server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
