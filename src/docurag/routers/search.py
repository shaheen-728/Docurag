import logging
from contextlib import nullcontext
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from google.genai import errors as google_errors
from openai import APIError as OpenAIAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from docurag.core.rate_limit import limiter
from docurag.core.tracing import get_tracing_client
from docurag.db.session import get_session
from docurag.retrieval.cache import cached_search
from docurag.retrieval.reranker import RerankerUnavailable
from docurag.schemas.search import SearchRequest, SearchResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Search"])

SessionDependency = Annotated[AsyncSession, Depends(get_session)]


async def execute_search(
    payload: SearchRequest,
    session: AsyncSession,
    *,
    debug: bool,
) -> SearchResponse:
    client = get_tracing_client()

    context = (
        client.start_as_current_observation(
            name="retrieval",
            as_type="retriever",
            metadata={
                "mode": payload.mode,
                "top_k": payload.top_k,
                "candidate_k": payload.candidate_k,
                "rerank_requested": payload.rerank,
                "cache_requested": payload.use_cache,
                "debug": debug,
                "query_characters": len(payload.query),
            },
        )
        if client is not None
        else nullcontext()
    )

    with context as observation:
        try:
            result = await cached_search(session, payload, debug=debug)

            if observation is not None:
                observation.update(
                    output={
                        "cache_hit": result.cache_hit,
                        "candidate_count": result.candidate_count,
                        "result_count": len(result.results),
                        "chunk_ids": [str(hit.chunk_id) for hit in result.results],
                        "latency_ms": result.latency_ms,
                        "rerank_applied": result.rerank_applied,
                    }
                )

            return result

        except (
            google_errors.APIError,
            OpenAIAPIError,
            httpx.TransportError,
        ) as error:
            logger.warning(
                "search_provider_failed error_type=%s",
                type(error).__name__,
            )

            raise HTTPException(
                status_code=503,
                detail=(
                    "Query embedding is unavailable. Try keyword mode or retry later."
                ),
            ) from error

        except RerankerUnavailable as error:
            logger.exception("search_reranker_failed")

            raise HTTPException(
                status_code=503,
                detail=(
                    "Reranker unavailable. Check API logs or retry with rerank=false."
                ),
            ) from error


@router.post("/search", response_model=SearchResponse)
@limiter.limit("30/minute")
async def search_documents(
    request: Request,
    payload: SearchRequest,
    session: SessionDependency,
) -> SearchResponse:
    return await execute_search(payload, session, debug=False)


@router.get("/search", response_model=SearchResponse)
@limiter.limit("30/minute")
async def debug_search(
    request: Request,
    payload: Annotated[SearchRequest, Query()],
    session: SessionDependency,
) -> SearchResponse:
    return await execute_search(payload, session, debug=True)
