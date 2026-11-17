from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from math import isfinite, sqrt
from typing import Literal

EmbeddingTask = Literal["document", "query"]


class LLMProvider(ABC):
    @property
    @abstractmethod
    def embedding_space(self) -> str:
        """Identify the provider, embedding model and dimensions."""
        ...

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        *,
        task: EmbeddingTask = "document",
    ) -> list[list[float]]:
        """Return one embedding per input text, in the same order."""
        ...

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Return a complete generated response."""
        ...

    @abstractmethod
    def stream(self, prompt: str) -> AsyncIterator[str]:
        """Yield generated text fragments."""
        ...

    @abstractmethod
    async def aclose(self) -> None:
        """Release client connections."""
        ...


def validate_texts(texts: list[str]) -> None:
    if any(not value.strip() for value in texts):
        raise ValueError("Embedding inputs cannot be blank.")


def normalize_embeddings(
    vectors: list[list[float]],
    *,
    count: int,
    dimensions: int,
) -> list[list[float]]:
    if len(vectors) != count:
        raise RuntimeError("Provider returned an unexpected embedding count.")

    normalized = []

    for vector in vectors:
        if len(vector) != dimensions:
            raise RuntimeError("Provider returned unexpected dimensions.")

        if not all(isfinite(value) for value in vector):
            raise RuntimeError("Provider returned non-finite values.")

        norm = sqrt(sum(value * value for value in vector))

        if norm == 0:
            raise RuntimeError("Provider returned a zero embedding.")

        normalized.append([value / norm for value in vector])

    return normalized
