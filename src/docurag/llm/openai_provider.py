from collections.abc import AsyncIterator
from contextlib import aclosing

from google.genai import types
from openai import AsyncOpenAI

from docurag.core.config import Settings
from docurag.llm.base import (
    EmbeddingTask,
    LLMProvider,
    normalize_embeddings,
    validate_texts,
)


class OpenAIProvider(LLMProvider):
    def __init__(self, settings: Settings) -> None:
        key = settings.openai_api_key.get_secret_value()

        if not key:
            raise ValueError("OPENAI_API_KEY is required.")

        self.settings = settings
        self.client = AsyncOpenAI(
            api_key=key,
            timeout=30.0,
            max_retries=2,
        )

    @property
    def embedding_space(self) -> str:
        model = self.settings.openai_embedding_model
        dimensions = self.settings.embedding_dimensions
        return f"openai:{model}:{dimensions}"

    async def embed(
        self,
        texts: list[str],
        *,
        task: EmbeddingTask = "document",
    ) -> list[list[float]]:
        if not texts:
            return []

        validate_texts(texts)

        # This API does not use a retrieval task-type parameter.
        response = await self.client.embeddings.create(
            model=self.settings.openai_embedding_model,
            input=texts,
            dimensions=self.settings.embedding_dimensions,
        )

        vectors = [
            item.embedding
            for item in sorted(response.data, key=lambda item: item.index)
        ]

        return normalize_embeddings(
            vectors,
            count=len(texts),
            dimensions=self.settings.embedding_dimensions,
        )

    async def generate(self, prompt: str) -> str:
        response = await self.client.responses.create(
            model=self.settings.openai_generation_model,
            input=prompt,
        )

        if not response.output_text:
            raise RuntimeError("OpenAI returned no text.")

        return response.output_text

    async def stream(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
    ) -> AsyncIterator[str]:
        response = await self.client.models.generate_content_stream(
            model=self.settings.gemini_generation_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
            ),
        )

        async with aclosing(response):
            async for chunk in response:
                if chunk.text:
                    yield chunk.text

    async def aclose(self) -> None:
        await self.client.close()
