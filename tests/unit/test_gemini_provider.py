import asyncio
from contextlib import aclosing
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from docurag.core.config import Settings
from docurag.llm.gemini import GeminiProvider


@pytest.fixture(params=[False, True], ids=["tracing-off", "tracing-on"])
def provider_setup(request):
    settings = Settings(
        _env_file=None,
        postgres_password="test",
        gemini_api_key="test",
    )

    models = SimpleNamespace(
        embed_content=AsyncMock(),
        generate_content=AsyncMock(),
        generate_content_stream=AsyncMock(),
    )
    sdk = SimpleNamespace(
        models=models,
        aclose=AsyncMock(),
    )
    tracing = MagicMock() if request.param else None

    with (
        patch(
            "docurag.llm.gemini.genai.Client",
            return_value=SimpleNamespace(aio=sdk),
        ),
        patch(
            "docurag.llm.gemini.get_tracing_client",
            return_value=tracing,
        ),
        patch("docurag.llm.gemini.record_generation") as generation,
        patch("docurag.llm.gemini.record_embedding") as embedding,
    ):
        provider = GeminiProvider(settings)
        yield provider, sdk, generation, embedding


def test_embedding_normalization_and_query_task(provider_setup):
    provider, sdk, _, embedding = provider_setup

    sdk.models.embed_content.return_value = SimpleNamespace(
        embeddings=[SimpleNamespace(values=[3.0, 4.0] + [0.0] * 766)]
    )

    vectors = asyncio.run(provider.embed(["test question"], task="query"))

    assert len(vectors) == 1
    assert len(vectors[0]) == 768
    assert vectors[0][:2] == pytest.approx([0.6, 0.8])

    arguments = sdk.models.embed_content.await_args.kwargs
    assert arguments["config"].task_type == "RETRIEVAL_QUERY"
    assert arguments["config"].output_dimensionality == 768

    embedding.assert_called_once()
    assert embedding.call_args.kwargs["completed"] is True


def test_generate_forwards_rules_and_records_usage(provider_setup):
    provider, sdk, generation, _ = provider_setup

    sdk.models.generate_content.return_value = SimpleNamespace(
        text="Test answer",
        usage_metadata=None,
    )

    answer = asyncio.run(
        provider.generate(
            "Test question",
            system_instruction="Test rules",
        )
    )

    assert answer == "Test answer"

    arguments = sdk.models.generate_content.await_args.kwargs
    assert arguments["config"].system_instruction == "Test rules"

    generation.assert_called_once()
    assert generation.call_args.kwargs["operation"] == "generate"
    assert generation.call_args.kwargs["completed"] is True


def test_stream_yields_fragments_and_closes_response(provider_setup):
    provider, sdk, generation, _ = provider_setup
    closed = []

    async def response():
        try:
            for fragment in ("Hello", " world"):
                yield SimpleNamespace(
                    text=fragment,
                    usage_metadata=None,
                )
        finally:
            closed.append(True)

    sdk.models.generate_content_stream.return_value = response()

    async def collect():
        return [
            fragment
            async for fragment in provider.stream(
                "Test question",
                system_instruction="Test rules",
            )
        ]

    assert asyncio.run(collect()) == ["Hello", " world"]
    assert closed == [True]

    arguments = sdk.models.generate_content_stream.await_args.kwargs
    assert arguments["config"].system_instruction == "Test rules"

    generation.assert_called_once()
    assert generation.call_args.kwargs["completed"] is True


def test_early_stream_close_records_incomplete_call(provider_setup):
    provider, sdk, generation, _ = provider_setup
    closed = []

    async def response():
        try:
            yield SimpleNamespace(text="Partial", usage_metadata=None)
            yield SimpleNamespace(text=" remainder", usage_metadata=None)
        finally:
            closed.append(True)

    sdk.models.generate_content_stream.return_value = response()

    async def consume_one():
        async with aclosing(provider.stream("Test question")) as stream:
            assert await anext(stream) == "Partial"

    asyncio.run(consume_one())

    assert closed == [True]
    generation.assert_called_once()
    assert generation.call_args.kwargs["completed"] is False
