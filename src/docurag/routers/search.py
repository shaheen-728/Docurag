import logging
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from google.genai import errors as google_errors
from openai import APIError as OpenAIAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from docurag.db.session import get_session
from docurag.retrieval.reranker import RerankerUnavailable
from docurag.retrieval.service import search
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
    try:
        return await search(session, payload, debug=debug)

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
            detail=("Query embedding is unavailable. Try keyword mode or retry later."),
        ) from error

    except RerankerUnavailable as error:
        logger.exception("search_reranker_failed")

        raise HTTPException(
            status_code=503,
            detail=("Reranker unavailable. Check API logs or retry with rerank=false."),
        ) from error


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    payload: SearchRequest,
    session: SessionDependency,
) -> SearchResponse:
    return await execute_search(payload, session, debug=False)


@router.get("/search", response_model=SearchResponse)
async def debug_search(
    payload: Annotated[SearchRequest, Query()],
    session: SessionDependency,
) -> SearchResponse:
    return await execute_search(payload, session, debug=True)
