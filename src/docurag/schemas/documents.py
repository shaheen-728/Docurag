from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class DocumentStatus(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    status: Literal["queued", "processing", "completed", "failed"]
    chunk_count: int
    processing_seconds: float | None
    error_message: str | None
    created_at: datetime


class DocumentUploadResponse(DocumentStatus):
    duplicate: bool = False
