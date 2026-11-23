import asyncio
import logging
from time import perf_counter

import httpx
from google.genai import errors as google_errors
from openai import APIConnectionError, APIStatusError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from docurag.core.config import get_settings
from docurag.llm.base import LLMProvider

logger = logging.getLogger(__name__)

RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}


def is_retryable(error: BaseException) -> bool:
    if isinstance(error, google_errors.APIError):
        if error.code == 429:
            message = str(error).casefold()

            # Daily quota cannot recover within this retry window.
            daily_quota_markers = (
                "perday",
                "per day",
                "daily quota",
            )

            if any(marker in message for marker in daily_quota_markers):
                return False

        return error.code in RETRYABLE_CODES

    if isinstance(error, APIStatusError):
        return error.status_code in RETRYABLE_CODES

    return isinstance(
        error,
        (APIConnectionError, httpx.TransportError),
    )


@retry(
    retry=retry_if_exception(is_retryable),
    stop=stop_after_attempt(4),
    wait=wait_random_exponential(
        multiplier=2,
        min=10,
        max=60,
    ),
    reraise=True,
)
async def embed_batch(
    provider: LLMProvider,
    texts: list[str],
) -> list[list[float]]:
    settings = get_settings()

    # Conservatively pace by input count, including retried batches.
    delay = max(
        2.0,
        60.0 * len(texts) / settings.embedding_texts_per_minute,
    )

    logger.info(
        "embedding_throttle count=%d wait_seconds=%.1f",
        len(texts),
        delay,
    )

    await asyncio.sleep(delay)

    return await provider.embed(texts, task="document")


async def embed_texts(
    provider: LLMProvider,
    texts: list[str],
    *,
    batch_size: int,
    document_id: str,
) -> list[list[float]]:
    if batch_size < 1:
        raise ValueError("Embedding batch size must be positive.")

    vectors: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        started = perf_counter()

        result = await embed_batch(provider, batch)

        if len(result) != len(batch):
            raise RuntimeError("Embedding count does not match chunk count.")

        vectors.extend(result)

        logger.info(
            "embedding_batch document_id=%s batch=%d count=%d seconds=%.3f",
            document_id,
            start // batch_size + 1,
            len(batch),
            perf_counter() - started,
        )

    return vectors
