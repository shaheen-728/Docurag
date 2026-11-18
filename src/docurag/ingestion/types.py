from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedSection:
    content: str
    source_file: str
    page_number: int | None = None
    section_heading: str | None = None


@dataclass(frozen=True)
class TextChunk:
    content: str
    chunk_index: int
    metadata: dict[str, Any]
