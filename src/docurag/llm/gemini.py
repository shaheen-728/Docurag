import logging
from collections.abc import AsyncIterator
from contextlib import aclosing, nullcontext
from time import perf_counter

from google import genai
from google.genai import types

from docurag.core.config import Settings
from docurag.core.tracing import get_tracing_client
from docurag.llm.accounting import (
    current_accounting,
    record_embedding,
    record_generation,
)
from docurag.llm.base import (
    EmbeddingTask,
    LLMProvider,
    normalize_embeddings,
    validate_texts,
)
from docurag.llm.costs import estimate_gemini_cost

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    def __init__(self, settings: Settings) -> None:
        key = settings.gemini_api_key.get_secret_value()

        if not key:
            raise ValueError("GEMINI_API_KEY is required.")

        self.settings = settings
        self.client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=30_000),
        ).aio

    @property
    def embedding_space(self) -> str:
        model = self.settings.gemini_embedding_model
        dimensions = self.settings.embedding_dimensions
        return f"gemini:{model}:{dimensions}"

    async def embed(
        self,
        texts: list[str],
        *,
        task: EmbeddingTask = "document",
    ) -> list[list[float]]:
        if not texts:
            return []

        client = get_tracing_client()

        context = (
            client.start_as_current_observation(
                name=f"gemini.embed.{task}",
                as_type="embedding",
                model=self.settings.gemini_embedding_model,
                metadata={
                    "provider": "gemini",
                    "task": task,
                    "text_count": len(texts),
                    "dimensions": self.settings.embedding_dimensions,
                    "usage_source": "not_returned_by_provider",
                },
            )
            if client is not None
            else nullcontext()
        )

        with context as observation:
            vectors = await self._embed(texts, task=task)

            if observation is not None:
                observation.update(
                    output={
                        "vector_count": len(vectors),
                        "dimensions": self.settings.embedding_dimensions,
                        "completed": True,
                    }
                )

            return vectors

    async def _embed(
        self,
        texts: list[str],
        *,
        task: EmbeddingTask = "document",
    ) -> list[list[float]]:
        validate_texts(texts)

        started = perf_counter()
        completed = False
        task_type = "RETRIEVAL_QUERY" if task == "query" else "RETRIEVAL_DOCUMENT"

        try:
            response = await self.client.models.embed_content(
                model=self.settings.gemini_embedding_model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.settings.embedding_dimensions,
                ),
            )

            vectors = [
                list(item.values)
                for item in response.embeddings or []
                if item.values is not None
            ]

            normalized = normalize_embeddings(
                vectors,
                count=len(texts),
                dimensions=self.settings.embedding_dimensions,
            )

            completed = True
            return normalized

        finally:
            latency_ms = (perf_counter() - started) * 1000

            record_embedding(
                model=self.settings.gemini_embedding_model,
                task=task,
                text_count=len(texts),
                latency_ms=latency_ms,
                completed=completed,
            )

            accounting = current_accounting.get()

            logger.info(
                "embedding_usage request_id=%s model=%s task=%s "
                "text_count=%s completed=%s latency_ms=%.3f "
                "input_tokens=unknown estimated_cost_usd=unknown",
                accounting.request_id if accounting else None,
                self.settings.gemini_embedding_model,
                task,
                len(texts),
                completed,
                latency_ms,
            )

    async def generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
    ) -> str:
        client = get_tracing_client()

        context = (
            client.start_as_current_observation(
                name="gemini.generate",
                as_type="generation",
                model=self.settings.gemini_generation_model,
                metadata={
                    "provider": "gemini",
                    "streaming": False,
                    "prompt_characters": len(prompt),
                },
            )
            if client is not None
            else nullcontext()
        )

        with context as observation:
            answer = await self._generate(
                prompt,
                system_instruction=system_instruction,
            )

            if observation is not None:
                observation.update(
                    output={
                        "answer_characters": len(answer),
                        "completed": True,
                    }
                )

            return answer

    async def _generate(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
    ) -> str:
        started = perf_counter()
        usage = None
        completed = False

        try:
            response = await self.client.models.generate_content(
                model=self.settings.gemini_generation_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                ),
            )

            usage = response.usage_metadata

            if not response.text:
                raise RuntimeError("Gemini returned no text.")

            completed = True
            return response.text

        finally:
            self._log_generation_usage(
                operation="generate",
                usage=usage,
                started=started,
                completed=completed,
            )

    async def stream(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        client = get_tracing_client()

        context = (
            client.start_as_current_observation(
                name="gemini.stream",
                as_type="generation",
                model=self.settings.gemini_generation_model,
                metadata={
                    "provider": "gemini",
                    "streaming": True,
                    "prompt_characters": len(prompt),
                },
            )
            if client is not None
            else nullcontext()
        )

        with context as observation:
            output_characters = 0
            completed = False

            try:
                async with aclosing(
                    self._stream(
                        prompt,
                        system_instruction=system_instruction,
                    )
                ) as stream:
                    async for fragment in stream:
                        output_characters += len(fragment)
                        yield fragment

                completed = True

            finally:
                if observation is not None:
                    observation.update(
                        output={
                            "answer_characters": output_characters,
                            "completed": completed,
                        }
                    )

    async def _stream(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        started = perf_counter()
        usage = None
        completed = False

        try:
            response = await self.client.models.generate_content_stream(
                model=self.settings.gemini_generation_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                ),
            )

            async with aclosing(response):
                async for chunk in response:
                    if chunk.usage_metadata is not None:
                        usage = chunk.usage_metadata

                    if chunk.text:
                        yield chunk.text

            completed = True

        finally:
            self._log_generation_usage(
                operation="stream",
                usage=usage,
                started=started,
                completed=completed,
            )

    async def aclose(self) -> None:
        await self.client.aclose()

    def _log_generation_usage(
        self,
        *,
        operation: str,
        usage: types.GenerateContentResponseUsageMetadata | None,
        started: float,
        completed: bool,
    ) -> None:
        cost = estimate_gemini_cost(
            self.settings.gemini_generation_model,
            usage,
            completed=completed,
        )

        record_generation(
            provider="gemini",
            model=self.settings.gemini_generation_model,
            operation=operation,
            input_tokens=usage.prompt_token_count if usage else None,
            total_tokens=usage.total_token_count if usage else None,
            estimated_cost_usd=cost,
            completed=completed,
        )

        accounting = current_accounting.get()
        request_id = accounting.request_id if accounting else None

        logger.info(
            "llm_usage request_id=%s provider=gemini model=%s operation=%s "
            "completed=%s input_tokens=%s output_tokens=%s "
            "thinking_tokens=%s cached_input_tokens=%s "
            "total_tokens=%s latency_ms=%.3f "
            "estimated_cost_usd=%s cost_basis=standard_paid_equivalent",
            request_id,
            self.settings.gemini_generation_model,
            operation,
            completed,
            usage.prompt_token_count if usage else None,
            usage.candidates_token_count if usage else None,
            usage.thoughts_token_count if usage else None,
            usage.cached_content_token_count if usage else None,
            usage.total_token_count if usage else None,
            (perf_counter() - started) * 1000,
            cost,
        )
