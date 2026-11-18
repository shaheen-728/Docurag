import asyncio
import logging
from pathlib import Path
from time import perf_counter
from uuid import UUID

from sqlalchemy import delete, text
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from docurag.core.config import get_settings
from docurag.db.models import Chunk, Document
from docurag.ingestion.chunkers import chunk_sections
from docurag.ingestion.embeddings import embed_texts
from docurag.ingestion.parsers import parse_document
from docurag.llm.factory import build_provider
from docurag.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="docurag.ingest")
def ingest_document(document_id: str) -> None:
    asyncio.run(run_ingestion(UUID(document_id)))


async def run_ingestion(document_id: UUID) -> None:
    settings = get_settings()

    # Each task owns its engine within its asyncio.run() event loop.
    engine = create_async_engine(
        settings.database_url,
        poolclass=NullPool,
    )

    sessions = async_sessionmaker(
        engine,
        expire_on_commit=False,
    )

    try:
        # Hold an advisory lock on a separate connection while processing.
        # Duplicate tasks for this document cannot run simultaneously.
        async with engine.begin() as lock_connection:
            lock_key = document_id.int % (2**63)

            acquired = await lock_connection.scalar(
                text("SELECT pg_try_advisory_xact_lock(:key)"),
                {"key": lock_key},
            )

            if not acquired:
                logger.info(
                    "ingestion_already_running document_id=%s",
                    document_id,
                )
                return

            async with sessions.begin() as session:
                document = await session.get(Document, document_id)

                if document is None:
                    raise ValueError("Document does not exist.")

                if document.status == "completed":
                    logger.info(
                        "ingestion_already_completed document_id=%s",
                        document_id,
                    )
                    return

                filename = document.filename
                source_path = document.storage_path

                document.status = "processing"
                document.error_message = None
                document.processing_seconds = None

            started = perf_counter()
            provider = None

            try:
                if not source_path:
                    raise ValueError("Document has no saved source file.")

                path = Path(source_path).resolve()
                upload_root = Path(settings.upload_directory).resolve()

                if not path.is_relative_to(upload_root):
                    raise ValueError("Source file is outside upload storage.")

                if not path.is_file():
                    raise FileNotFoundError("Saved source file is missing.")

                sections = await asyncio.to_thread(
                    parse_document,
                    path,
                    source_file=filename,
                )

                chunks = await asyncio.to_thread(
                    chunk_sections,
                    sections,
                    settings,
                )

                if not chunks:
                    raise ValueError("Document produced no text chunks.")

                logger.info(
                    "ingestion_parsed document_id=%s sections=%d "
                    "chunks=%d seconds=%.3f",
                    document_id,
                    len(sections),
                    len(chunks),
                    perf_counter() - started,
                )

                provider = build_provider(settings)

                vectors = await embed_texts(
                    provider,
                    [chunk.content for chunk in chunks],
                    batch_size=settings.embedding_batch_size,
                    document_id=str(document_id),
                )

                # Replace chunks and mark completion in one transaction.
                async with sessions.begin() as session:
                    document = await session.get(Document, document_id)

                    if document is None:
                        raise ValueError("Document no longer exists.")

                    await session.execute(
                        delete(Chunk).where(Chunk.document_id == document_id)
                    )

                    session.add_all(
                        [
                            Chunk(
                                document_id=document_id,
                                chunk_index=chunk.chunk_index,
                                content=chunk.content,
                                embedding=vector,
                                embedding_model=provider.embedding_space,
                                metadata_=chunk.metadata,
                            )
                            for chunk, vector in zip(
                                chunks,
                                vectors,
                                strict=True,
                            )
                        ]
                    )

                    # Flush inserts before recording elapsed processing time.
                    await session.flush()

                    document.status = "completed"
                    document.chunk_count = len(chunks)
                    document.processing_seconds = perf_counter() - started
                    document.error_message = None

                logger.info(
                    "ingestion_completed document_id=%s chunks=%d seconds=%.3f",
                    document_id,
                    len(chunks),
                    perf_counter() - started,
                )

            except Exception as error:
                async with sessions.begin() as session:
                    document = await session.get(Document, document_id)

                    if document is not None:
                        document.status = "failed"
                        document.processing_seconds = perf_counter() - started
                        document.error_message = (
                            f"Ingestion failed: {type(error).__name__}. "
                            "Check worker logs."
                        )

                logger.exception(
                    "ingestion_failed document_id=%s",
                    document_id,
                )
                raise

            finally:
                if provider is not None:
                    try:
                        await provider.aclose()
                    except Exception:
                        logger.warning(
                            "provider_cleanup_failed document_id=%s",
                            document_id,
                        )

    finally:
        await engine.dispose()
