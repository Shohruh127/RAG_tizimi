"""
Retrieval layer for the Time-Aware Hybrid RAG System.

Components:
- Query analyser  – extracts date constraints and intent from a free-text query.
- Dense search    – Qdrant vector search with time-travel payload filter.
- Sparse search   – Elasticsearch BM25 with time-travel date filter.
- Hybrid fusion   – Reciprocal Rank Fusion (RRF) to merge both result lists.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from elasticsearch import AsyncElasticsearch
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    DatetimeRange,
    FieldCondition,
    Filter,
    MatchValue,
    Range,
)

from database import ELASTICSEARCH_INDEX, QDRANT_COLLECTION
from ingestion import embed_text
from models import ChunkResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM client (OpenAI-compatible; supports Gemini via openai-compatible endpoint)
# ---------------------------------------------------------------------------

_openai_client: Optional[AsyncOpenAI] = None


def get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
    return _openai_client


LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Query analyser
# ---------------------------------------------------------------------------

_DATE_PATTERNS: list[Tuple[str, str]] = [
    # ISO date: 2022-05-10
    (r"\b(\d{4}-\d{2}-\d{2})\b", "%Y-%m-%d"),
    # "in 2022" / "2022-yil"
    (r"\bin\s+(\d{4})\b", "%Y"),
    (r"\b(\d{4})-yil\b", "%Y"),
    # "may 2022" / "2022 may"
    (r"\b([A-Za-z]+ \d{4})\b", "%B %Y"),
    (r"\b(\d{4} [A-Za-z]+)\b", "%Y %B"),
]


@dataclass
class QueryAnalysis:
    clean_query: str
    point_in_time: Optional[datetime.date] = None
    intent: str = "search"


def analyse_query(query: str) -> QueryAnalysis:
    """
    Extract optional date constraints from *query* and return a QueryAnalysis.

    The date, if found, is treated as an exact point-in-time for time-travel search.
    When the extracted pattern is a year only, the date is set to 31 Dec of that year
    so that the entire year is covered.
    """
    pit: Optional[datetime.date] = None

    for pattern, fmt in _DATE_PATTERNS:
        m = re.search(pattern, query, re.IGNORECASE)
        if m:
            raw = m.group(1)
            try:
                parsed = datetime.datetime.strptime(raw, fmt).date()
                # For year-only matches, use Dec 31 so the full year is included.
                if fmt == "%Y":
                    parsed = parsed.replace(month=12, day=31)
                pit = parsed
                break
            except ValueError:
                continue

    return QueryAnalysis(
        clean_query=query.strip(),
        point_in_time=pit,
    )


# ---------------------------------------------------------------------------
# LLM-powered query extractor (optional – used for structured intent)
# ---------------------------------------------------------------------------

async def extract_query_with_llm(query: str) -> QueryAnalysis:
    """
    Use the LLM to extract structured information from the query.
    Falls back to regex-based extraction if the LLM call fails.
    """
    try:
        client = get_openai_client()
        system_prompt = (
            "You are a query analyser for a legal document search system. "
            "Given a user query, extract:\n"
            "1. A clean search query (without date references).\n"
            "2. A specific date in YYYY-MM-DD format if the query refers to a point in time, "
            "otherwise null.\n"
            "Respond ONLY with a JSON object: "
            '{"clean_query": "...", "point_in_time": "YYYY-MM-DD or null"}'
        )
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=128,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        pit_raw = data.get("point_in_time")
        pit = (
            datetime.datetime.strptime(pit_raw, "%Y-%m-%d").date()
            if pit_raw and pit_raw != "null"
            else None
        )
        return QueryAnalysis(clean_query=data.get("clean_query", query), point_in_time=pit)
    except Exception as exc:
        logger.warning("LLM query extraction failed (%s); falling back to regex.", exc)
        return analyse_query(query)


# ---------------------------------------------------------------------------
# Qdrant (dense) search
# ---------------------------------------------------------------------------

async def dense_search(
    qdrant: AsyncQdrantClient,
    query: str,
    point_in_time: Optional[datetime.date],
    top_k: int,
) -> List[dict]:
    """
    Perform a vector similarity search in Qdrant with time-travel filtering.

    Returns a list of payload dicts (with an added ``_score`` key).
    """
    vector = embed_text(query)

    # --- Build the payload filter ---
    if point_in_time is not None:
        pit_ts = float(
            datetime.datetime.combine(point_in_time, datetime.time.min).timestamp()
        )
        # valid_from_ts <= pit_ts  AND  (valid_to_ts >= pit_ts OR valid_to_ts IS NULL)
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="valid_from_ts",
                    range=Range(lte=pit_ts),
                ),
            ],
            should=[
                FieldCondition(
                    key="valid_to_ts",
                    range=Range(gte=pit_ts),
                ),
                # NULL valid_to means the chunk is still current
                FieldCondition(
                    key="is_current",
                    match=MatchValue(value=True),
                ),
            ],
            minimum_should_match=1,
        )
    else:
        qdrant_filter = Filter(
            must=[
                FieldCondition(key="is_current", match=MatchValue(value=True))
            ]
        )

    results = await qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=vector,
        query_filter=qdrant_filter,
        limit=top_k,
        with_payload=True,
    )

    hits = []
    for hit in results:
        payload = dict(hit.payload or {})
        payload["_score"] = hit.score
        payload["_id"] = str(hit.id)
        hits.append(payload)
    return hits


# ---------------------------------------------------------------------------
# Elasticsearch (sparse) search
# ---------------------------------------------------------------------------

async def sparse_search(
    es: AsyncElasticsearch,
    query: str,
    point_in_time: Optional[datetime.date],
    top_k: int,
) -> List[dict]:
    """
    Perform a BM25 keyword search in Elasticsearch with time-travel filtering.

    Returns a list of source dicts (with an added ``_score`` key).
    """
    if point_in_time is not None:
        pit_str = point_in_time.isoformat()
        date_filter: dict = {
            "bool": {
                "must": [
                    {"range": {"valid_from": {"lte": pit_str}}},
                ],
                "should": [
                    {"range": {"valid_to": {"gte": pit_str}}},
                    # Documents where valid_to is null stored as "9999-12-31"
                    {"range": {"valid_to": {"gte": "9999-12-31"}}},
                ],
                "minimum_should_match": 1,
            }
        }
    else:
        date_filter = {"term": {"is_current": True}}

    body = {
        "query": {
            "bool": {
                "must": [{"match": {"text": {"query": query}}}],
                "filter": [date_filter],
            }
        },
        "size": top_k,
    }

    response = await es.search(index=ELASTICSEARCH_INDEX, body=body)
    hits = []
    for hit in response["hits"]["hits"]:
        src = dict(hit["_source"])
        src["_score"] = hit["_score"]
        src["_id"] = hit["_id"]
        hits.append(src)
    return hits


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

RRF_K: int = 60  # standard constant


def reciprocal_rank_fusion(
    *ranked_lists: List[dict],
    id_key: str = "chunk_id",
) -> List[Tuple[str, float]]:
    """
    Merge multiple ranked result lists using RRF.

    Returns a list of (chunk_id, rrf_score) tuples sorted by descending score.
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            cid = item.get(id_key) or item.get("_id", "")
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Full hybrid retrieval pipeline
# ---------------------------------------------------------------------------

async def hybrid_retrieve(
    qdrant: AsyncQdrantClient,
    es: AsyncElasticsearch,
    query: str,
    top_k: int = 5,
    use_llm_extractor: bool = False,
) -> Tuple[List[ChunkResult], Optional[datetime.date]]:
    """
    Run the full hybrid retrieval pipeline and return the top-k chunks.

    Steps:
    1. Analyse query (extract date / intent).
    2. Parallel dense + sparse search.
    3. RRF fusion.
    4. Fetch full chunk data for top-k unique IDs.
    """
    # Step 1: Query analysis
    if use_llm_extractor:
        analysis = await extract_query_with_llm(query)
    else:
        analysis = analyse_query(query)

    pit = analysis.point_in_time
    logger.info("Query analysis: clean='%s', pit=%s", analysis.clean_query, pit)

    # Step 2: Parallel search
    dense_task = dense_search(qdrant, analysis.clean_query, pit, top_k * 2)
    sparse_task = sparse_search(es, analysis.clean_query, pit, top_k * 2)
    dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

    logger.info("Dense hits: %d, Sparse hits: %d", len(dense_results), len(sparse_results))

    # Step 3: RRF
    fused = reciprocal_rank_fusion(dense_results, sparse_results)
    top_ids_scores = fused[:top_k]

    # Step 4: Build ChunkResult list using the raw payloads we already have
    id_to_payload: Dict[str, dict] = {}
    for item in dense_results + sparse_results:
        cid = item.get("chunk_id") or item.get("_id", "")
        if cid and cid not in id_to_payload:
            id_to_payload[cid] = item

    chunks: List[ChunkResult] = []
    for chunk_id, rrf_score in top_ids_scores:
        payload = id_to_payload.get(chunk_id)
        if payload is None:
            continue

        # Parse dates (Qdrant stores as timestamps, ES as ISO strings)
        def _to_date(val) -> Optional[datetime.date]:
            if val is None:
                return None
            if isinstance(val, (int, float)):
                return datetime.datetime.fromtimestamp(val).date()
            try:
                return datetime.date.fromisoformat(str(val))
            except ValueError:
                return None

        valid_from_raw = payload.get("valid_from") or payload.get("valid_from_ts")
        valid_to_raw = payload.get("valid_to") or payload.get("valid_to_ts")

        valid_from = _to_date(valid_from_raw) or datetime.date.today()
        valid_to = _to_date(valid_to_raw)

        chunks.append(
            ChunkResult(
                chunk_id=chunk_id,
                doc_id=str(payload.get("doc_id", "")),
                text=str(payload.get("text", "")),
                valid_from=valid_from,
                valid_to=valid_to,
                is_current=bool(payload.get("is_current", False)),
                score=rrf_score,
            )
        )

    return chunks, pit


# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------

async def generate_answer(
    query: str,
    chunks: List[ChunkResult],
    point_in_time: Optional[datetime.date],
) -> str:
    """
    Pass the retrieved chunks to the LLM and return a cited answer.
    """
    if not chunks:
        return "Kechirasiz, so'rovingizga mos hujjat topilmadi."

    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        date_info = f"valid_from={chunk.valid_from}"
        if chunk.valid_to:
            date_info += f", valid_to={chunk.valid_to}"
        context_parts.append(f"[{i}] (chunk_id={chunk.chunk_id}, {date_info})\n{chunk.text}")

    context = "\n\n".join(context_parts)

    pit_note = (
        f"The user is asking about the law as it stood on {point_in_time}."
        if point_in_time
        else "The user is asking about current law."
    )

    system_prompt = (
        "You are a legal expert assistant specialising in Uzbek law. "
        "Answer the user's question accurately and concisely based ONLY on the provided context. "
        "Cite the relevant chunk numbers in square brackets, e.g. [1], [2]. "
        f"{pit_note}"
    )
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    try:
        client = get_openai_client()
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("LLM generation failed: %s", exc)
        # Return a basic fallback with the raw chunks
        return (
            "LLM generation failed. Top retrieved chunks:\n"
            + "\n".join(f"[{i}] {c.text[:200]}" for i, c in enumerate(chunks, 1))
        )
