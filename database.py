"""
Database layer for the Time-Aware Hybrid RAG System.

Responsibilities:
- PostgreSQL: async engine + session factory (SQLAlchemy 2.0 / asyncpg)
- Elasticsearch 8.x: async client + index bootstrap
- Qdrant: async client + collection bootstrap
- ``init_all`` helper called at application startup
"""

from __future__ import annotations

import logging
import os
from typing import AsyncGenerator

from elasticsearch import AsyncElasticsearch
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    PayloadSchemaType,
    VectorParams,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from models import Base

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (read from environment variables with sensible defaults)
# ---------------------------------------------------------------------------

POSTGRES_DSN: str = os.getenv(
    "POSTGRES_DSN",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_db",
)

ELASTICSEARCH_URL: str = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ELASTICSEARCH_INDEX: str = os.getenv("ELASTICSEARCH_INDEX", "legal_chunks")

QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "legal_chunks")
QDRANT_VECTOR_DIM: int = int(os.getenv("QDRANT_VECTOR_DIM", "1024"))

# ---------------------------------------------------------------------------
# PostgreSQL (SQLAlchemy 2.0 async)
# ---------------------------------------------------------------------------

engine: AsyncEngine = create_async_engine(
    POSTGRES_DSN,
    echo=False,
    pool_pre_ping=True,
)

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async SQLAlchemy session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_postgres() -> None:
    """Create all tables if they do not already exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("PostgreSQL tables initialised.")


# ---------------------------------------------------------------------------
# Elasticsearch 8.x
# ---------------------------------------------------------------------------

ES_INDEX_MAPPING: dict = {
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "text": {
                "type": "text",
                "analyzer": "standard",
            },
            "valid_from": {"type": "date", "format": "yyyy-MM-dd"},
            "valid_to": {"type": "date", "format": "yyyy-MM-dd", "null_value": "9999-12-31"},
            "is_current": {"type": "boolean"},
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
}

es_client: AsyncElasticsearch = AsyncElasticsearch(
    hosts=[ELASTICSEARCH_URL],
    request_timeout=30,
)


async def init_elasticsearch() -> None:
    """Create the Elasticsearch index if it does not already exist."""
    exists = await es_client.indices.exists(index=ELASTICSEARCH_INDEX)
    if not exists:
        await es_client.indices.create(index=ELASTICSEARCH_INDEX, body=ES_INDEX_MAPPING)
        logger.info("Elasticsearch index '%s' created.", ELASTICSEARCH_INDEX)
    else:
        logger.info("Elasticsearch index '%s' already exists.", ELASTICSEARCH_INDEX)


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

qdrant_client: AsyncQdrantClient = AsyncQdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
)


async def init_qdrant() -> None:
    """Create the Qdrant collection if it does not already exist, and ensure payload indices."""
    collections = await qdrant_client.get_collections()
    existing = {c.name for c in collections.collections}

    if QDRANT_COLLECTION not in existing:
        await qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=QDRANT_VECTOR_DIM,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Qdrant collection '%s' created.", QDRANT_COLLECTION)
    else:
        logger.info("Qdrant collection '%s' already exists.", QDRANT_COLLECTION)

    # Create payload indices for fast filtering
    for field, schema_type in [
        ("is_current", PayloadSchemaType.BOOL),
        ("valid_from_ts", PayloadSchemaType.FLOAT),
        ("valid_to_ts", PayloadSchemaType.FLOAT),
    ]:
        try:
            await qdrant_client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field,
                field_schema=schema_type,
            )
        except Exception:
            # Index may already exist; ignore
            pass


# ---------------------------------------------------------------------------
# Unified initialisation
# ---------------------------------------------------------------------------

async def init_all() -> None:
    """Initialise all storage backends. Call this once at application startup."""
    await init_postgres()
    await init_elasticsearch()
    await init_qdrant()
    logger.info("All storage backends initialised.")


async def close_all() -> None:
    """Close all storage backend connections gracefully."""
    await engine.dispose()
    await es_client.close()
    await qdrant_client.close()
    logger.info("All storage backend connections closed.")
