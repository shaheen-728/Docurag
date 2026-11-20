from time import perf_counter

from sqlalchemy import func, literal_column, select
from sqlalchemy.ext.asyncio import AsyncSession

from docurag.core.config import get_settings
from docurag.db.models import Chunk, Document
from docurag.llm.factory import build_provider
from docurag.retrieval.reranker import rerank_hits
from docurag.retrieval.rrf import reciprocal_rank_fusion
from docurag.schemas.search import (
    SearchHit,
    SearchRequest,
    SearchResponse,
)


def document_filters(request: SearchRequest) -> list:
    filters = [Document.status == "completed"]

    if request.source is not None:
        filters.append(Document.filename == request.source)

    if request.created_after is not None:
        filters.append(Document.created_at >= request.created_after)

    if request.created_before is not None:
        filters.append(Document.created_at < request.created_before)

    return filters


def make_hit(chunk: Chunk, filename: str) -> SearchHit:
    return SearchHit(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        filename=filename,
        content=chunk.content,
        metadata=chunk.metadata_,
    )


async def keyword_search(
    session: AsyncSession,
    request: SearchRequest,
) -> list[SearchHit]:
    tsquery = func.websearch_to_tsquery(
        literal_column("'english'::regconfig"),
        request.query,
    )

    score = func.ts_rank(
        Chunk.search_vector,
        tsquery,
    ).label("keyword_score")

    statement = (
        select(Chunk, Document.filename, score)
        .join(Document, Document.id == Chunk.document_id)
        .where(
            *document_filters(request),
            Chunk.search_vector.op("@@")(tsquery),
        )
        .order_by(score.desc(), Chunk.id)
        .limit(request.candidate_k)
    )

    rows = (await session.execute(statement)).all()
    hits = []

    for rank, (chunk, filename, keyword_score) in enumerate(
        rows,
        start=1,
    ):
        hit = make_hit(chunk, filename)
        hit.keyword_rank = rank
        hit.keyword_score = float(keyword_score)
        hits.append(hit)

    return hits


async def vector_search(
    session: AsyncSession,
    request: SearchRequest,
    embedding: list[float],
    embedding_space: str,
) -> list[SearchHit]:
    distance = Chunk.embedding.cosine_distance(embedding)

    statement = (
        select(
            Chunk,
            Document.filename,
            distance.label("distance"),
        )
        .join(Document, Document.id == Chunk.document_id)
        .where(
            *document_filters(request),
            Chunk.embedding_model == embedding_space,
        )
        .order_by(distance)
        .limit(request.candidate_k)
    )

    rows = (await session.execute(statement)).all()
    hits = []

    for rank, (chunk, filename, value) in enumerate(rows, start=1):
        hit = make_hit(chunk, filename)
        hit.vector_rank = rank
        hit.vector_distance = float(value)
        hit.vector_similarity = 1.0 - float(value)
        hits.append(hit)

    return hits


async def search(
    session: AsyncSession,
    request: SearchRequest,
    *,
    debug: bool = False,
) -> SearchResponse:
    started = perf_counter()
    settings = get_settings()

    vector_hits: list[SearchHit] = []
    keyword_hits: list[SearchHit] = []

    if request.mode in {"vector", "hybrid"}:
        provider = build_provider(settings)

        try:
            embeddings = await provider.embed(
                [request.query],
                task="query",
            )

            if len(embeddings) != 1:
                raise RuntimeError("Expected one query embedding.")

            embedding = embeddings[0]
            embedding_space = provider.embedding_space
        finally:
            await provider.aclose()

        vector_hits = await vector_search(
            session,
            request,
            embedding,
            embedding_space,
        )

    if request.mode in {"keyword", "hybrid"}:
        keyword_hits = await keyword_search(session, request)

    # Merge candidates by chunk ID while preserving both methods' scores.
    combined = {hit.chunk_id: hit for hit in vector_hits}

    for hit in keyword_hits:
        if hit.chunk_id in combined:
            existing = combined[hit.chunk_id]
            existing.keyword_rank = hit.keyword_rank
            existing.keyword_score = hit.keyword_score
        else:
            combined[hit.chunk_id] = hit

    fusion_scores = reciprocal_rank_fusion(
        [
            [hit.chunk_id for hit in vector_hits],
            [hit.chunk_id for hit in keyword_hits],
        ]
    )

    for chunk_id, hit in combined.items():
        hit.rrf_score = fusion_scores[chunk_id]

    ranked = sorted(
        combined.values(),
        key=lambda hit: (-hit.rrf_score, str(hit.chunk_id)),
    )

    results = ranked[: request.top_k]
    debug_candidates = ranked

    rerank_applied = False
    rerank_ms = 0.0

    if request.rerank and ranked:
        rerank_started = perf_counter()

        shortlist_size = max(
            settings.reranker_candidates,
            request.top_k,
        )

        reranked = await rerank_hits(
            request.query,
            ranked[:shortlist_size],
        )

        rerank_ms = round(
            (perf_counter() - rerank_started) * 1000,
            3,
        )

        rerank_applied = True
        results = reranked[: request.top_k]

        # Keep debug candidates in RRF order, adding reranker scores
        # and ranks only to candidates that were rescored.
        scored_by_id = {hit.chunk_id: hit for hit in reranked}

        debug_candidates = [scored_by_id.get(hit.chunk_id, hit) for hit in ranked]

    return SearchResponse(
        query=request.query,
        mode=request.mode,
        candidate_count=len(ranked),
        latency_ms=round(
            (perf_counter() - started) * 1000,
            3,
        ),
        results=results,
        candidates=debug_candidates if debug else None,
        rerank_applied=rerank_applied,
        rerank_ms=rerank_ms,
        reranker_model=(settings.reranker_model if rerank_applied else None),
    )
