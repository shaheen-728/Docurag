import asyncio
import os
from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from docurag.core.config import get_settings
from docurag.db.models import EMBEDDING_DIMENSIONS, Chunk, Document

pytestmark = pytest.mark.skipif(
    os.getenv("DOCURAG_RUN_DB_TESTS") != "1",
    reason="Database integration tests require explicit opt-in.",
)


def test_document_chunk_storage_search_and_cascade():
    asyncio.run(check_database())


async def check_database():
    settings = get_settings()

    if settings.postgres_db != "docurag_test":
        raise RuntimeError("Integration tests must use the docurag_test database.")

    engine = create_async_engine(settings.database_url)

    try:
        async with engine.connect() as connection:
            transaction = await connection.begin()

            try:
                database = await connection.scalar(text("SELECT current_database()"))
                assert database == "docurag_test"

                async with AsyncSession(bind=connection) as session:
                    document_id = uuid4()
                    chunk_id = uuid4()
                    embedding = [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)

                    session.add(
                        Document(
                            id=document_id,
                            filename="integration-policy.txt",
                            content_type="text/plain",
                            content_hash=uuid4().hex * 2,
                            status="completed",
                            chunk_count=1,
                        )
                    )
                    await session.flush()

                    session.add(
                        Chunk(
                            id=chunk_id,
                            document_id=document_id,
                            chunk_index=0,
                            content=(
                                "Employees receive annual leave with manager approval."
                            ),
                            embedding=embedding,
                            embedding_model="integration-test:768",
                            metadata_={
                                "source_file": "integration-policy.txt",
                                "page_number": 2,
                                "section_heading": "Annual Leave",
                            },
                        )
                    )
                    await session.flush()

                    # JSONB metadata survives storage.
                    page = await session.scalar(
                        select(Chunk.metadata_["page_number"].as_integer()).where(
                            Chunk.id == chunk_id
                        )
                    )
                    assert page == 2

                    # pgvector stores the expected dimensions.
                    dimensions = await session.scalar(
                        select(func.vector_dims(Chunk.embedding)).where(
                            Chunk.id == chunk_id
                        )
                    )
                    assert dimensions == EMBEDDING_DIMENSIONS

                    # An identical vector has near-zero cosine distance.
                    distance = await session.scalar(
                        select(Chunk.embedding.cosine_distance(embedding)).where(
                            Chunk.id == chunk_id
                        )
                    )
                    assert distance == pytest.approx(0.0, abs=1e-6)

                    # The generated tsvector supports keyword matching.
                    matched_id = await session.scalar(
                        select(Chunk.id).where(
                            Chunk.id == chunk_id,
                            Chunk.search_vector.op("@@")(
                                func.plainto_tsquery(
                                    "english",
                                    "annual leave",
                                )
                            ),
                        )
                    )
                    assert matched_id == chunk_id

                    # Database-level deletion cascades to child chunks.
                    await session.execute(
                        delete(Document).where(Document.id == document_id)
                    )

                    remaining = await session.scalar(
                        select(func.count())
                        .select_from(Chunk)
                        .where(Chunk.document_id == document_id)
                    )
                    assert remaining == 0

            finally:
                # Leave no test records behind, including on failure.
                await transaction.rollback()

    finally:
        await engine.dispose()
