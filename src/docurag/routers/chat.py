import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from contextlib import aclosing
from time import perf_counter

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from docurag.chat.generation import stream_answer
from docurag.chat.prompts import (
    ANSWER_RULES,
    INSUFFICIENT_CONTEXT,
    build_answer_prompt,
)
from docurag.chat.rewrite import rewrite_query
from docurag.routers.search import SessionDependency, execute_search
from docurag.schemas.chat import ChatRequest
from docurag.schemas.search import SearchRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def sse(event: str, payload: dict) -> str:
    """Encode one SSE event with a JSON payload."""
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


@router.post("/chat")
async def chat_with_documents(
    payload: ChatRequest,
    session: SessionDependency,
) -> StreamingResponse:
    started = perf_counter()

    try:
        standalone_query = await rewrite_query(payload)
    except TimeoutError as error:
        raise HTTPException(
            status_code=504,
            detail="Follow-up query rewriting timed out.",
        ) from error
    except Exception as error:
        logger.warning(
            "chat_rewrite_failed error_type=%s",
            type(error).__name__,
        )

        if isinstance(error, ValidationError):
            detail = "Query rewriting returned an invalid search query."
        else:
            detail = "Follow-up query rewriting is unavailable."

        raise HTTPException(
            status_code=503,
            detail=detail,
        ) from error

    try:
        async with asyncio.timeout(30):
            retrieved = await execute_search(
                SearchRequest(
                    query=standalone_query,
                    mode="vector",
                    top_k=5,
                    candidate_k=20,
                    source=payload.source,
                    rerank=False,
                ),
                session,
                debug=False,
            )
    except TimeoutError as error:
        raise HTTPException(
            status_code=504,
            detail="Document retrieval timed out.",
        ) from error

    # Release the read transaction before waiting on generation.
    await session.rollback()

    prompt, available_citations = build_answer_prompt(
        standalone_query,
        retrieved.results,
    )

    async def events() -> AsyncIterator[str]:
        outcome = "error"

        try:
            if not retrieved.results:
                yield sse("token", {"text": INSUFFICIENT_CONTEXT})
                yield sse("citations", {"items": []})
                outcome = "completed"
                yield sse("done", {"status": "completed"})
                return

            fragments: list[str] = []
            output_characters = 0

            # Overall budget covers primary and fallback generation.
            async with asyncio.timeout(65):
                async with aclosing(
                    stream_answer(
                        prompt,
                        system_instruction=ANSWER_RULES,
                    )
                ) as stream:
                    async for fragment in stream:
                        output_characters += len(fragment)

                        if output_characters > 20000:
                            raise RuntimeError(
                                "Generated answer exceeded output limit."
                            )

                        fragments.append(fragment)
                        yield sse("token", {"text": fragment})

            answer = "".join(fragments)

            if not answer.strip():
                raise RuntimeError("Generation returned no answer.")

            used_numbers = {int(value) for value in CITATION_PATTERN.findall(answer)}
            available_numbers = {citation.number for citation in available_citations}

            if not used_numbers <= available_numbers:
                yield sse(
                    "error",
                    {
                        "code": "invalid_citations",
                        "message": (
                            "The answer referenced an unavailable source. "
                            "Treat the streamed answer as incomplete."
                        ),
                    },
                )
                return

            if not used_numbers and answer.strip() != INSUFFICIENT_CONTEXT:
                yield sse(
                    "error",
                    {
                        "code": "missing_citations",
                        "message": (
                            "The answer did not include source citations. "
                            "Treat the streamed answer as incomplete."
                        ),
                    },
                )
                return

            cited_sources = [
                citation.model_dump(mode="json")
                for citation in available_citations
                if citation.number in used_numbers
            ]

            yield sse("citations", {"items": cited_sources})
            outcome = "completed"
            yield sse("done", {"status": "completed"})

        except TimeoutError:
            yield sse(
                "error",
                {
                    "code": "generation_timeout",
                    "message": "Answer generation timed out.",
                },
            )

        except asyncio.CancelledError:
            outcome = "cancelled"
            raise

        except Exception:
            logger.exception("chat_failed")

            yield sse(
                "error",
                {
                    "code": "generation_failed",
                    "message": "Answer generation failed. Please retry.",
                },
            )

        finally:
            logger.info(
                "chat_finished status=%s latency_ms=%.3f",
                outcome,
                (perf_counter() - started) * 1000,
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
