from typing import Literal

from pydantic import BaseModel, Field, model_validator

from docurag.schemas.search import SearchText


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: SearchText


class ChatRequest(BaseModel):
    query: SearchText
    history: list[ChatMessage] = Field(default_factory=list, max_length=12)
    source: str | None = Field(default=None, max_length=255)

    @model_validator(mode="after")
    def validate_history_size(self) -> "ChatRequest":
        total = sum(len(message.content) for message in self.history)

        if total > 8000:
            raise ValueError("Combined history must not exceed 8000 characters.")

        return self


class Citation(BaseModel):
    number: int
    chunk_id: str
    document_id: str
    document: str
    page: int | None = None
    heading: str | None = None
    snippet: str
