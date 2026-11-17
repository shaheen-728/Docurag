from collections.abc import AsyncIterator
from contextlib import aclosing

from google import genai
from google.genai import types

from docurag.core.config import Settings
from docurag.llm.base import (
    EmbeddingTask,
    LLMProvider,
    normalize_embeddings,
    validate_texts,
)


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

        validate_texts(texts)

        task_type = "RETRIEVAL_QUERY" if task == "query" else "RETRIEVAL_DOCUMENT"

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

        return normalize_embeddings(
            vectors,
            count=len(texts),
            dimensions=self.settings.embedding_dimensions,
        )

    async def generate(self, prompt: str) -> str:
        response = await self.client.models.generate_content(
            model=self.settings.gemini_generation_model,
            contents=prompt,
        )

        if not response.text:
            raise RuntimeError("Gemini returned no text.")

        return response.text

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        response = await self.client.models.generate_content_stream(
            model=self.settings.gemini_generation_model,
            contents=prompt,
        )

        async with aclosing(response):
            async for chunk in response:
                if chunk.text:
                    yield chunk.text

    async def aclose(self) -> None:
        await self.client.aclose()
