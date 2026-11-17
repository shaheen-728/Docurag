from collections.abc import AsyncIterator

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

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        async with self.client.responses.stream(
            model=self.settings.openai_generation_model,
            input=prompt,
        ) as response:
            async for event in response:
                if event.type == "response.output_text.delta":
                    yield event.delta
                elif event.type == "error":
                    raise RuntimeError("OpenAI streaming request failed.")
                elif event.type in {
                    "response.failed",
                    "response.incomplete",
                }:
                    raise RuntimeError("OpenAI stream did not complete successfully.")

    async def aclose(self) -> None:
        await self.client.close()
