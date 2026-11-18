import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from docurag.core.config import Settings
from docurag.ingestion.types import ParsedSection, TextChunk

TOKEN_ENCODING = "cl100k_base"


def chunk_sections(
    sections: list[ParsedSection],
    settings: Settings,
) -> list[TextChunk]:
    if settings.chunker == "recursive":
        size = settings.character_chunk_size
        overlap = settings.character_chunk_overlap
    else:
        size = settings.token_chunk_size
        overlap = settings.token_chunk_overlap

    if overlap >= size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    options = {
        "chunk_size": size,
        "chunk_overlap": overlap,
        "separators": ["\n\n", "\n", " ", ""],
        "keep_separator": True,
        "strip_whitespace": True,
    }

    if settings.chunker == "recursive":
        splitter = RecursiveCharacterTextSplitter(**options)
    else:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=TOKEN_ENCODING,
            allowed_special=set(),
            disallowed_special=(),
            **options,
        )

    encoding = tiktoken.get_encoding(TOKEN_ENCODING)
    chunks: list[TextChunk] = []

    for section in sections:
        for content in splitter.split_text(section.content):
            if not content.strip():
                continue

            token_count = len(encoding.encode(content, disallowed_special=()))

            chunks.append(
                TextChunk(
                    content=content,
                    chunk_index=len(chunks),
                    metadata={
                        "source_file": section.source_file,
                        "page_number": section.page_number,
                        "section_heading": section.section_heading,
                        "chunker": settings.chunker,
                        "chunk_size": size,
                        "chunk_overlap": overlap,
                        "token_encoding": TOKEN_ENCODING,
                        "token_count": token_count,
                    },
                )
            )

    return chunks
