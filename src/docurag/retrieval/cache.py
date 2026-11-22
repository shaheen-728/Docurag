import asyncio
import hashlib
import json
import logging
from time import perf_counter

from pydantic import ValidationError
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from docurag.core.config import Settings, get_settings
from docurag.db.models import CorpusRevision
from docurag.retrieval.service import search
from docurag.schemas.search import SearchRequest, SearchResponse

logger = logging.getLogger(__name__)

CACHE_VERSION = "v1"


def build_cache_key(
    request: SearchRequest,
    settings: Settings,
    revision: int,
) -> str:
    identity = {
        "version": CACHE_VERSION,
        "database": {
            "host": settings.db_host,
            "port": settings.db_port,
            "name": settings.postgres_db,
            "user": settings.postgres_user,
        },
        "corpus_revision": revision,
        "request": request.model_dump(
            mode="json",
            exclude={"use_cache"},
        ),
        "embedding": {
            "provider": settings.llm_provider,
            "model": getattr(
                settings,
                f"{settings.llm_provider}_embedding_model",
            ),
            "dimensions": settings.embedding_dimensions,
        },
        "reranker": {
            "model": settings.reranker_model,
            "candidates": settings.reranker_candidates,
        },
    }

    serialized = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"docurag:retrieval:{CACHE_VERSION}:{digest}"


async def read_cached_response(
    settings: Settings,
    key: str,
) -> SearchResponse | None:
    try:
        async with asyncio.timeout(0.5):
            async with Redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=0.2,
                socket_timeout=0.2,
            ) as client:
                raw = await client.get(key)

        if raw is None:
            return None

        return SearchResponse.model_validate_json(raw)

    except (RedisError, TimeoutError, ValidationError) as error:
        logger.warning(
            "retrieval_cache_read_failed error_type=%s",
            type(error).__name__,
        )
        return None


async def write_cached_response(
    settings: Settings,
    key: str,
    response: SearchResponse,
) -> None:
    try:
        async with asyncio.timeout(0.5):
            async with Redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=0.2,
                socket_timeout=0.2,
            ) as client:
                await client.set(
                    key,
                    response.model_dump_json(),
                    ex=settings.retrieval_cache_ttl_seconds,
                )

    except (RedisError, TimeoutError) as error:
        logger.warning(
            "retrieval_cache_write_failed error_type=%s",
            type(error).__name__,
        )


async def cached_search(
    session: AsyncSession,
    request: SearchRequest,
    *,
    debug: bool = False,
) -> SearchResponse:
    settings = get_settings()

    # Debug searches and evaluation requests bypass both reads and writes.
    if debug or not request.use_cache or not settings.retrieval_cache_enabled:
        return await search(session, request, debug=debug)

    started = perf_counter()

    revision = await session.scalar(
        select(CorpusRevision.revision).where(CorpusRevision.id == 1)
    )

    if revision is None:
        raise RuntimeError("Corpus revision row is missing.")

    key = build_cache_key(request, settings, revision)
    cached = await read_cached_response(settings, key)

    if cached is not None:
        logger.info("retrieval_cache_hit")

        return cached.model_copy(
            update={
                "cache_hit": True,
                "latency_ms": round(
                    (perf_counter() - started) * 1000,
                    3,
                ),
                # No reranking work ran during this request.
                "rerank_ms": 0.0,
            }
        )

    logger.info("retrieval_cache_miss")

    response = await search(session, request, debug=False)

    # Do not populate the cache if the corpus changed during retrieval.
    latest_revision = await session.scalar(
        select(CorpusRevision.revision).where(CorpusRevision.id == 1)
    )

    if latest_revision == revision:
        await write_cached_response(settings, key, response)
    else:
        logger.info("retrieval_cache_write_skipped corpus_changed=True")

    return response.model_copy(
        update={
            "cache_hit": False,
            "latency_ms": round(
                (perf_counter() - started) * 1000,
                3,
            ),
        }
    )
