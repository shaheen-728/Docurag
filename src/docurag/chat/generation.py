import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing

import httpx
from google.genai import errors as google_errors
from openai import APIConnectionError, APIStatusError

from docurag.core.config import get_settings
from docurag.llm.factory import build_provider

logger = logging.getLogger(__name__)


def is_transient(error: Exception) -> bool:
    """Only retry failures that may succeed with another attempt."""
    if isinstance(
        error,
        (TimeoutError, httpx.TransportError, APIConnectionError),
    ):
        return True

    if isinstance(error, google_errors.APIError):
        return error.code in {408, 429, 500, 502, 503, 504}

    if isinstance(error, APIStatusError):
        return error.status_code in {408, 429, 500, 502, 503, 504}

    return False


async def stream_answer(
    prompt: str,
    *,
    system_instruction: str,
) -> AsyncIterator[str]:
    settings = get_settings()

    model_field = (
        "gemini_generation_model"
        if settings.llm_provider == "gemini"
        else "openai_generation_model"
    )

    primary_model = getattr(settings, model_field)
    models = [primary_model]

    fallback = settings.generation_fallback_model

    if fallback and fallback.strip() and fallback != primary_model:
        models.append(fallback.strip())

    for attempt, model in enumerate(models, start=1):
        attempt_settings = settings.model_copy(
            update={model_field: model},
        )
        provider = build_provider(attempt_settings)
        emitted_text = False

        try:
            async with asyncio.timeout(settings.generation_attempt_timeout_seconds):
                async with aclosing(
                    provider.stream(
                        prompt,
                        system_instruction=system_instruction,
                    )
                ) as stream:
                    async for fragment in stream:
                        if not fragment:
                            continue

                        emitted_text = True
                        yield fragment

            if not emitted_text:
                raise RuntimeError("Generation returned no text.")

            return

        except Exception as error:
            can_fallback = (
                not emitted_text and attempt < len(models) and is_transient(error)
            )

            logger.warning(
                "generation_attempt_failed "
                "model=%s attempt=%s error_type=%s fallback=%s",
                model,
                attempt,
                type(error).__name__,
                can_fallback,
            )

            if not can_fallback:
                raise

        finally:
            await provider.aclose()
