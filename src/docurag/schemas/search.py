from typing import Annotated, Any, Literal, Self
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    Field,
    StringConstraints,
    model_validator,
)

SearchText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=1000,
    ),
]


class SearchRequest(BaseModel):
    query: SearchText

    # Keyword mode works without an external embedding request.
    mode: Literal["keyword", "vector", "hybrid"] = "keyword"

    top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=20, ge=1, le=100)

    source: str | None = Field(default=None, max_length=255)

    created_after: AwareDatetime | None = None
    created_before: AwareDatetime | None = None

    rerank: bool = False
    use_cache: bool = False

    @model_validator(mode="after")
    def validate_options(self) -> Self:
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be at least top_k.")

        if (
            self.created_after is not None
            and self.created_before is not None
            and self.created_after >= self.created_before
        ):
            raise ValueError("created_after must be earlier than created_before.")

        if self.rerank and self.top_k > 20:
            raise ValueError("Reranked results are limited to 20.")
        return self


class SearchHit(BaseModel):
    chunk_id: UUID
    document_id: UUID
    filename: str
    content: str
    metadata: dict[str, Any]

    vector_distance: float | None = None
    vector_similarity: float | None = None
    vector_rank: int | None = None

    keyword_score: float | None = None
    keyword_rank: int | None = None

    rrf_score: float = 0.0

    rerank_score: float | None = None
    rerank_rank: int | None = None


class SearchResponse(BaseModel):
    query: str
    mode: str
    candidate_count: int
    latency_ms: float

    results: list[SearchHit]

    # Included by the debug GET endpoint.
    candidates: list[SearchHit] | None = None

    rerank_applied: bool = False
    rerank_ms: float = 0.0
    reranker_model: str | None = None
    cache_hit: bool = False
