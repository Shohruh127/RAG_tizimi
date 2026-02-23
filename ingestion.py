"""
Ingestion pipeline for the Time-Aware Hybrid RAG System.

Workflow per chunk:
1. Expire the current version in PostgreSQL (SCD Type 2).
2. Insert a new PostgreSQL row.
3. Mark the old ES document as not current and index the new one.
4. Upsert the vector into Qdrant.

All steps are attempted; errors are logged without aborting the whole batch.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import uuid
from typing import Optional

from elasticsearch import NotFoundError
from qdrant_client.http.models import PointStruct
from sentence_transformers import SentenceTransformer
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database import (
    ELASTICSEARCH_INDEX,
    QDRANT_COLLECTION,
    es_client,
    qdrant_client,
)
from models import ChunkInput, ChunkORM, DocumentInput, DocumentORM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding model (loaded once at module import time)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-large"
_embedder: Optional[SentenceTransformer] = None


def get_embedder() -> SentenceTransformer:
    """Return (and lazily initialise) the shared SentenceTransformer model."""
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model '%s' …", EMBEDDING_MODEL_NAME)
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded.")
    return _embedder


def embed_text(text: str) -> list[float]:
    """Return the embedding vector for *text*."""
    model = get_embedder()
    # multilingual-e5-large works best with a query/passage prefix
    embedding = model.encode(f"passage: {text}", normalize_embeddings=True)
    return embedding.tolist()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _date_to_ts(d: Optional[datetime.date]) -> Optional[float]:
    """Convert a date (or None) to a Unix timestamp (float) for Qdrant payload."""
    if d is None:
        return None
    return float(datetime.datetime.combine(d, datetime.time.min).timestamp())


# ---------------------------------------------------------------------------
# Core ingestion functions
# ---------------------------------------------------------------------------

async def _expire_current_chunk(
    session: AsyncSession,
    doc_id: uuid.UUID,
    chunk_input: ChunkInput,
    expire_date: datetime.date,
) -> Optional[uuid.UUID]:
    """
    Find the current chunk with matching *hierarchical_context* for *doc_id* and
    expire it (SCD Type 2: set valid_to = expire_date, is_current = False).

    Returns the old chunk_id if one was found, otherwise None.

    Note: if *chunk_input.hierarchical_context* is ``None`` the function returns
    ``None`` immediately without touching the database (backward-compatibility
    mode – chunks ingested without a structural identifier cannot be expired by
    this mechanism).
    """
    if chunk_input.hierarchical_context is None:
        return None

    stmt = (
        select(ChunkORM)
        .where(
            ChunkORM.doc_id == doc_id,
            ChunkORM.hierarchical_context == chunk_input.hierarchical_context,
            ChunkORM.is_current.is_(True),
        )
        .limit(1)
    )
    result = await session.execute(stmt)
    existing = result.scalar_one_or_none()
    if existing is None:
        return None

    await session.execute(
        update(ChunkORM)
        .where(ChunkORM.chunk_id == existing.chunk_id)
        .values(valid_to=expire_date, is_current=False)
    )
    return existing.chunk_id


async def _save_chunk_to_postgres(
    session: AsyncSession,
    doc_id: uuid.UUID,
    chunk: ChunkInput,
) -> ChunkORM:
    """Insert a new chunk row and return the ORM object (not yet flushed)."""
    orm = ChunkORM(
        doc_id=doc_id,
        text=chunk.text,
        hierarchical_context=chunk.hierarchical_context,
        valid_from=chunk.valid_from,
        valid_to=None,
        is_current=True,
    )
    session.add(orm)
    await session.flush()  # Populate chunk_id without committing
    return orm


async def _save_chunk_to_elasticsearch(chunk: ChunkORM) -> None:
    """Index or re-index a single chunk in Elasticsearch."""
    doc = {
        "chunk_id": str(chunk.chunk_id),
        "doc_id": str(chunk.doc_id),
        "text": chunk.text,
        "valid_from": chunk.valid_from.isoformat(),
        "valid_to": chunk.valid_to.isoformat() if chunk.valid_to else None,
        "is_current": chunk.is_current,
    }
    await es_client.index(
        index=ELASTICSEARCH_INDEX,
        id=str(chunk.chunk_id),
        document=doc,
    )


async def _expire_es_chunk(old_chunk_id: uuid.UUID, expire_date: datetime.date) -> None:
    """Mark an old ES document as not current and set its valid_to date."""
    try:
        await es_client.update(
            index=ELASTICSEARCH_INDEX,
            id=str(old_chunk_id),
            doc={
                "is_current": False,
                "valid_to": expire_date.isoformat(),
            },
        )
    except NotFoundError:
        logger.warning("ES document '%s' not found during expiry; skipping.", old_chunk_id)


async def _save_chunk_to_qdrant(chunk: ChunkORM, vector: list[float]) -> None:
    """Upsert a single chunk vector + payload into Qdrant."""
    payload = {
        "chunk_id": str(chunk.chunk_id),
        "doc_id": str(chunk.doc_id),
        "is_current": chunk.is_current,
        "valid_from_ts": _date_to_ts(chunk.valid_from),
        "valid_to_ts": _date_to_ts(chunk.valid_to),
    }
    point = PointStruct(
        id=str(chunk.chunk_id),
        vector=vector,
        payload=payload,
    )
    await qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[point],
    )


async def _expire_qdrant_chunk(old_chunk_id: uuid.UUID, expire_date: datetime.date) -> None:
    """Update Qdrant payload for an expired chunk."""
    await qdrant_client.set_payload(
        collection_name=QDRANT_COLLECTION,
        payload={
            "is_current": False,
            "valid_to_ts": _date_to_ts(expire_date),
        },
        points=[str(old_chunk_id)],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def ingest_document(session: AsyncSession, payload: DocumentInput) -> DocumentORM:
    """
    Ingest a complete document (metadata + chunks).

    For each chunk:
    - If an identical current chunk already exists (same doc + text), expire it first.
    - Write the new version to PostgreSQL, Elasticsearch, and Qdrant.

    The PostgreSQL session is committed by the caller (e.g. FastAPI dependency).
    """
    # ---- 1. Upsert the document record ---------------------------------
    stmt = select(DocumentORM).where(DocumentORM.title == payload.title).limit(1)
    result = await session.execute(stmt)
    doc_orm = result.scalar_one_or_none()

    if doc_orm is None:
        doc_orm = DocumentORM(title=payload.title, status=payload.status)
        session.add(doc_orm)
        await session.flush()
    else:
        doc_orm.status = payload.status
        await session.flush()

    today = datetime.date.today()

    # ---- 2. Process each chunk -----------------------------------------
    for chunk_input in payload.chunks:
        # --- SCD Type 2: expire existing version in PostgreSQL ----------
        old_pg_id: Optional[uuid.UUID] = await _expire_current_chunk(
            session, doc_orm.doc_id, chunk_input, today
        )

        # --- Insert new PostgreSQL row ----------------------------------
        try:
            new_chunk = await _save_chunk_to_postgres(session, doc_orm.doc_id, chunk_input)
        except Exception as exc:
            logger.error(
                "PostgreSQL insert failed for doc '%s', chunk text='%s…': %s",
                doc_orm.doc_id,
                chunk_input.text[:60],
                exc,
            )
            continue

        # --- Embed the text --------------------------------------------
        try:
            vector = await asyncio.to_thread(embed_text, chunk_input.text)
        except Exception as exc:
            logger.error(
                "Embedding failed for chunk '%s': %s", new_chunk.chunk_id, exc
            )
            continue

        # --- Elasticsearch -------------------------------------------
        try:
            if old_pg_id:
                await _expire_es_chunk(old_pg_id, today)
            await _save_chunk_to_elasticsearch(new_chunk)
        except Exception as exc:
            logger.error(
                "Elasticsearch operation failed for chunk '%s': %s",
                new_chunk.chunk_id,
                exc,
            )

        # --- Qdrant --------------------------------------------------
        try:
            if old_pg_id:
                await _expire_qdrant_chunk(old_pg_id, today)
            await _save_chunk_to_qdrant(new_chunk, vector)
        except Exception as exc:
            logger.error(
                "Qdrant operation failed for chunk '%s': %s",
                new_chunk.chunk_id,
                exc,
            )

    return doc_orm
