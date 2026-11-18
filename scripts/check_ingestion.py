from pathlib import Path

from docurag.core.config import get_settings
from docurag.ingestion.chunkers import chunk_sections
from docurag.ingestion.parsers import parse_document
from docurag.ingestion.types import ParsedSection

sections = parse_document(
    Path("data/samples/leavepolicy.md"),
    source_file="leavepolicy.md",
)

print("Parsed sections:", len(sections))

for section in sections:
    print("Heading:", section.section_heading)

for mode in ("recursive", "token"):
    settings = get_settings().model_copy(update={"chunker": mode})

    chunks = chunk_sections(sections, settings)

    print(f"\nChunker: {mode}")
    print("Chunks:", len(chunks))

    for chunk in chunks:
        assert chunk.content.strip()
        assert chunk.metadata["source_file"] == "leavepolicy.md"

        if mode == "token":
            assert chunk.metadata["token_count"] <= settings.token_chunk_size

        print("Index:", chunk.chunk_index)
        print("Metadata:", chunk.metadata)
        print("Content:", chunk.content[:100])


long_section = ParsedSection(
    content=(
        "Employees receive annual leave and must request approval. "
        "Unused leave is subject to the carry-forward policy.\n\n"
    )
    * 100,
    source_file="long-policy.md",
    section_heading="Annual Leave",
)

for mode in ("recursive", "token"):
    settings = get_settings().model_copy(update={"chunker": mode})

    chunks = chunk_sections([long_section], settings)

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.metadata["section_heading"] == "Annual Leave"

        if mode == "recursive":
            assert len(chunk.content) <= settings.character_chunk_size
        else:
            assert chunk.metadata["token_count"] <= settings.token_chunk_size

    print(f"{mode}: long section split into {len(chunks)} chunks")
    print("First chunk ending:", repr(chunks[0].content[-150:]))
    print("Second chunk beginning:", repr(chunks[1].content[:150]))
