import pytest
import tiktoken

from docurag.core.config import Settings
from docurag.ingestion.chunkers import chunk_sections
from docurag.ingestion.types import ParsedSection


def settings_for(mode: str) -> Settings:
    return Settings(
        _env_file=None,
        postgres_password="unit-test-unused",
        chunker=mode,
        character_chunk_size=160,
        character_chunk_overlap=30,
        token_chunk_size=64,
        token_chunk_overlap=12,
    )


@pytest.mark.parametrize("mode", ["recursive", "token"])
def test_long_section_is_split_with_limits_and_metadata(mode):
    settings = settings_for(mode)
    section = ParsedSection(
        content="Employees must obtain manager approval before leave. " * 100,
        source_file="policy.pdf",
        page_number=3,
        section_heading="Annual Leave",
    )

    chunks = chunk_sections([section], settings)
    encoding = tiktoken.get_encoding("cl100k_base")

    assert len(chunks) > 1
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))

    for chunk in chunks:
        actual_tokens = len(encoding.encode(chunk.content, disallowed_special=()))

        assert chunk.content.strip()
        assert chunk.metadata["source_file"] == "policy.pdf"
        assert chunk.metadata["page_number"] == 3
        assert chunk.metadata["section_heading"] == "Annual Leave"
        assert chunk.metadata["token_count"] == actual_tokens

        if mode == "recursive":
            assert len(chunk.content) <= 160
        else:
            assert actual_tokens <= 64


@pytest.mark.parametrize("mode", ["recursive", "token"])
def test_short_sections_preserve_content_and_boundaries(mode):
    sections = [
        ParsedSection(
            content="Annual leave requires approval.",
            source_file="policy.md",
            section_heading="Annual Leave",
        ),
        ParsedSection(
            content="Report sick leave before your shift.",
            source_file="policy.md",
            section_heading="Sick Leave",
        ),
    ]

    chunks = chunk_sections(sections, settings_for(mode))

    assert [chunk.content for chunk in chunks] == [
        section.content for section in sections
    ]
    assert [chunk.metadata["section_heading"] for chunk in chunks] == [
        "Annual Leave",
        "Sick Leave",
    ]


@pytest.mark.parametrize("mode", ["recursive", "token"])
def test_empty_input_returns_no_chunks(mode):
    assert chunk_sections([], settings_for(mode)) == []


@pytest.mark.parametrize("mode", ["recursive", "token"])
def test_overlap_must_be_smaller_than_size(mode):
    settings = settings_for(mode)
    settings = settings.model_copy(
        update={
            "character_chunk_overlap": 160,
            "token_chunk_overlap": 64,
        }
    )

    with pytest.raises(ValueError, match="overlap"):
        chunk_sections([], settings)


@pytest.mark.parametrize("mode", ["recursive", "token"])
def test_overlap_preserves_boundary_words_without_losing_content(mode):
    words = [f"item{number:04d}" for number in range(300)]
    section = ParsedSection(
        content=" ".join(words),
        source_file="overlap.txt",
    )

    chunks = chunk_sections([section], settings_for(mode))
    chunk_words = [chunk.content.split() for chunk in chunks]

    assert len(chunks) > 1

    # Every input word survives splitting.
    assert {word for values in chunk_words for word in values} == set(words)

    # Adjacent chunks share boundary words and make forward progress.
    for previous, current in zip(
        chunk_words,
        chunk_words[1:],
        strict=False,
    ):
        shared = set(previous) & set(current)
        assert shared, "Adjacent chunks lost their overlap."
        assert set(current) - set(previous), "Chunk made no progress."

        overlap_length = max(
            length
            for length in range(1, min(len(previous), len(current)) + 1)
            if previous[-length:] == current[:length]
        )
        assert overlap_length > 0
