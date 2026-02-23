"""
Data models for the Time-Aware Hybrid RAG System.

Includes:
- SQLAlchemy ORM models for PostgreSQL (documents, chunks with SCD Type 2)
- Pydantic schemas for API request/response validation
- Pydantic schemas for ingestion input
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field
from sqlalchemy import Boolean, Date, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ---------------------------------------------------------------------------
# SQLAlchemy ORM models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class DocumentORM(Base):
    """Represents a legal document (law, decree, etc.)."""

    __tablename__ = "documents"

    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="active")

    chunks: Mapped[List["ChunkORM"]] = relationship(
        "ChunkORM", back_populates="document", cascade="all, delete-orphan"
    )


class ChunkORM(Base):
    """
    Represents a versioned text chunk (SCD Type 2).

    When a chunk is amended:
      - The old row gets ``valid_to = current_date`` and ``is_current = False``.
      - A new row is inserted with ``valid_from = current_date``, ``valid_to = None``,
        and ``is_current = True``.
    """

    __tablename__ = "chunks"

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    doc_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("documents.doc_id"), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # Structural identifier (e.g. article number / section path) used for SCD
    # Type 2 matching: when a chunk is amended, the old row is found and expired
    # by matching this field rather than the text content (which changes).
    hierarchical_context: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    valid_from: Mapped[date] = mapped_column(Date, nullable=False)
    valid_to: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    document: Mapped["DocumentORM"] = relationship(
        "DocumentORM", back_populates="chunks"
    )


# ---------------------------------------------------------------------------
# Pydantic schemas – Ingestion
# ---------------------------------------------------------------------------

class ChunkInput(BaseModel):
    """A single chunk as provided in the ingestion JSON payload."""

    text: str = Field(..., description="The text content of the chunk.")
    valid_from: date = Field(..., description="The date this chunk version became effective.")
    hierarchical_context: Optional[str] = Field(
        None,
        description="Structural identifier (e.g. article number / section path) used for SCD Type 2 matching.",
    )


class DocumentInput(BaseModel):
    """Top-level ingestion payload for a single document."""

    title: str = Field(..., description="Human-readable title of the document.")
    status: str = Field("active", description="Document lifecycle status.")
    chunks: List[ChunkInput] = Field(..., description="Ordered list of text chunks.")


# ---------------------------------------------------------------------------
# Pydantic schemas – Retrieval / API
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    """Incoming search request body."""

    query: str = Field(..., min_length=1, description="Natural-language query string.")
    top_k: int = Field(5, ge=1, le=20, description="Number of top chunks to return.")


class ChunkResult(BaseModel):
    """A single retrieved chunk returned to the caller."""

    chunk_id: str
    doc_id: str
    text: str
    valid_from: date
    valid_to: Optional[date]
    is_current: bool
    score: float = Field(..., description="Combined RRF score.")


class SearchResponse(BaseModel):
    """Response body for the /api/v1/search endpoint."""

    query: str
    point_in_time: Optional[date] = Field(
        None,
        description="The date used for time-travel filtering, if extracted from the query.",
    )
    answer: str = Field(..., description="LLM-generated answer with citations.")
    chunks: List[ChunkResult] = Field(
        ..., description="Top-k retrieved chunks used as context."
    )


class IngestionResponse(BaseModel):
    """Response body for the document ingestion endpoint."""

    doc_id: str
    chunks_ingested: int
    message: str
