import argparse
import asyncio
import hashlib
import json
from collections import defaultdict
from math import isfinite
from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from docurag.core.config import Settings, get_settings
from docurag.db.models import EMBEDDING_DIMENSIONS, Chunk, Document
from docurag.ingestion.embeddings import embed_batch
from docurag.llm.factory import build_provider

EXPERIMENT_SIZES = (256, 512, 1024)


def checksum(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def validate_vector(vector: list[float]) -> None:
    if (
        len(vector) != EMBEDDING_DIMENSIONS
        or not all(isfinite(value) for value in vector)
        or not any(value != 0 for value in vector)
    ):
        raise ValueError("Invalid embedding vector.")


def configured_embedding_space(settings: Settings) -> str:
    model = getattr(
        settings,
        f"{settings.llm_provider}_embedding_model",
    )
    return f"{settings.llm_provider}:{model}:{settings.embedding_dimensions}"


async def load_reusable_vectors(
    settings: Settings,
    source_ids: set[UUID],
    embedding_space: str,
) -> dict[str, list[float]]:
    reusable: dict[str, list[float]] = {}

    databases = [
        settings.postgres_db,
        *[f"docurag_eval_token{size}" for size in EXPERIMENT_SIZES],
    ]

    for database in databases:
        source_settings = settings.model_copy(update={"postgres_db": database})
        source_engine = create_async_engine(source_settings.database_url)
        source_sessions = async_sessionmaker(source_engine)

        try:
            async with source_sessions() as session:
                if database == settings.postgres_db:
                    spaces = set(
                        (
                            await session.scalars(
                                select(Chunk.embedding_model)
                                .where(Chunk.document_id.in_(source_ids))
                                .distinct()
                            )
                        ).all()
                    )

                    if spaces != {embedding_space}:
                        raise ValueError(
                            "Configured embedding space does not match "
                            f"the baseline: {spaces}"
                        )

                rows = (
                    await session.execute(
                        select(Chunk.content, Chunk.embedding)
                        .where(
                            Chunk.document_id.in_(source_ids),
                            Chunk.embedding_model == embedding_space,
                        )
                        .order_by(Chunk.document_id, Chunk.chunk_index)
                    )
                ).all()

            added = 0

            for content, embedding in rows:
                if content in reusable:
                    continue

                vector = [float(value) for value in embedding]
                validate_vector(vector)
                reusable[content] = vector
                added += 1

            print(
                f"Reusable unique texts from {database}: {added}",
                flush=True,
            )
        finally:
            await source_engine.dispose()

    return reusable


async def update_document_progress(
    session: AsyncSession,
    document_id: UUID,
    expected_count: int,
) -> None:
    await session.flush()

    document = await session.get(Document, document_id)

    if document is None:
        raise ValueError("Experiment document disappeared.")

    stored_count = await session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
    )

    if stored_count is None or stored_count > expected_count:
        raise ValueError("Unexpected stored chunk count.")

    document.chunk_count = stored_count
    document.status = "completed" if stored_count == expected_count else "queued"
    document.error_message = None


async def run(args: argparse.Namespace) -> None:
    settings = get_settings()
    target_database = f"docurag_eval_token{args.size}"

    if settings.postgres_db.startswith("docurag_eval_"):
        raise ValueError("Run with the baseline database configured as POSTGRES_DB.")

    source_raw = (args.prepared / "sources.json").read_bytes()
    manifest_raw = (args.prepared / "manifest.json").read_bytes()
    chunk_raw = (args.prepared / f"token-{args.size}.json").read_bytes()

    sources = json.loads(source_raw)
    records = json.loads(chunk_raw)

    preparation_hash = checksum(manifest_raw + source_raw)
    chunk_file_hash = checksum(chunk_raw)
    embedding_space = configured_embedding_space(settings)

    source_ids = {UUID(item["document_id"]) for item in sources}
    grouped = defaultdict(list)
    expected = {}

    if not source_ids or len(source_ids) != len(sources):
        raise ValueError("Invalid source-document inventory.")

    source_by_id = {UUID(item["document_id"]): item for item in sources}

    for record in records:
        chunk_id = UUID(record["chunk_id"])
        document_id = UUID(record["document_id"])
        metadata = record["metadata"]

        if chunk_id in expected:
            raise ValueError("Duplicate prepared chunk ID.")

        if document_id not in source_ids:
            raise ValueError("Chunk references an unknown document.")

        if (
            metadata["chunker"] != "token"
            or metadata["chunk_size"] != args.size
            or metadata["chunk_overlap"] != 50
        ):
            raise ValueError("Chunk settings do not match this variant.")

        if not record["content"].strip():
            raise ValueError("Prepared chunk is empty.")

        expected[chunk_id] = record
        grouped[document_id].append(record)

    if set(grouped) != source_ids:
        raise ValueError("Every source document must have prepared chunks.")

    for document_records in grouped.values():
        document_records.sort(key=lambda item: item["chunk_index"])

        indexes = [item["chunk_index"] for item in document_records]

        if indexes != list(range(len(document_records))):
            raise ValueError("Chunk indexes must be contiguous.")

    target_settings = settings.model_copy(update={"postgres_db": target_database})
    target_engine = create_async_engine(target_settings.database_url)
    sessions = async_sessionmaker(
        target_engine,
        expire_on_commit=False,
    )
    provider = None

    try:
        async with sessions() as session:
            documents = (await session.scalars(select(Document))).all()
            stored = (await session.scalars(select(Chunk))).all()

        if {doc.id for doc in documents} != source_ids:
            raise ValueError("Target documents differ from the inventory.")

        for document in documents:
            source = source_by_id[document.id]

            if (
                document.filename != source["filename"]
                or document.content_hash != source["content_hash"]
            ):
                raise ValueError("Target document differs from the source inventory.")

            marker = document.metadata_.get("_chunk_experiment", {})

            if (
                marker.get("size_tokens") != args.size
                or marker.get("overlap_tokens") != 50
                or marker.get("preparation_sha256") != preparation_hash
            ):
                raise ValueError("Target experiment identity differs.")

            saved_hash = marker.get("chunk_file_sha256")

            if saved_hash is not None and saved_hash != chunk_file_hash:
                raise ValueError("Prepared chunk file changed.")

        existing = set()

        for chunk in stored:
            record = expected.get(chunk.id)

            if record is None:
                raise ValueError("Target contains an unexpected chunk.")

            if (
                chunk.document_id != UUID(record["document_id"])
                or chunk.chunk_index != record["chunk_index"]
                or chunk.content != record["content"]
                or chunk.metadata_ != record["metadata"]
            ):
                raise ValueError("Stored chunk differs from prepared data.")

            if chunk.embedding_model != embedding_space:
                raise ValueError("Stored experiment embedding model differs.")

            validate_vector([float(value) for value in chunk.embedding])
            existing.add(chunk.id)

        print(f"Target: {target_database}", flush=True)
        print(f"Prepared chunks: {len(records)}", flush=True)
        print(f"Already stored: {len(existing)}", flush=True)
        print(
            f"Missing: {len(records) - len(existing)}",
            flush=True,
        )

        reusable = await load_reusable_vectors(
            settings,
            source_ids,
            embedding_space,
        )

        missing_records = [
            record for record in records if UUID(record["chunk_id"]) not in existing
        ]

        reusable_count = sum(
            record["content"] in reusable for record in missing_records
        )
        new_texts = {
            record["content"]
            for record in missing_records
            if record["content"] not in reusable
        }

        print(
            f"Missing chunks reusable without API calls: {reusable_count}",
            flush=True,
        )
        print(
            f"New unique texts requiring API calls: {len(new_texts)}",
            flush=True,
        )

        if not args.execute:
            print("DRY RUN: no API calls or database writes.")
            return

        # Pin the exact chunk file before writing chunks.
        async with sessions() as session:
            async with session.begin():
                target_documents = (await session.scalars(select(Document))).all()

                for document in target_documents:
                    metadata = dict(document.metadata_)
                    marker = dict(metadata["_chunk_experiment"])
                    marker["chunk_file_sha256"] = chunk_file_hash
                    metadata["_chunk_experiment"] = marker
                    document.metadata_ = metadata

        batches = 0
        copied_chunks = 0
        embedded_texts = 0

        for document_id, document_records in grouped.items():
            missing = [
                record
                for record in document_records
                if UUID(record["chunk_id"]) not in existing
            ]

            if args.reuse_only:
                missing = [
                    record for record in missing if record["content"] in reusable
                ]

            for start in range(0, len(missing), args.batch_size):
                if args.max_batches is not None and batches >= args.max_batches:
                    print("Batch limit reached. Rerun to continue.")
                    return

                batch = missing[start : start + args.batch_size]

                # Exact-text reuse, with no whitespace normalization.
                batch_new_texts = list(
                    dict.fromkeys(
                        record["content"]
                        for record in batch
                        if record["content"] not in reusable
                    )
                )
                reused_in_batch = sum(record["content"] in reusable for record in batch)

                if batch_new_texts:
                    if args.reuse_only:
                        raise RuntimeError("Reuse-only mode attempted an API request.")

                    # Create the model client only when necessary.
                    if provider is None:
                        provider = build_provider(settings)

                        if provider.embedding_space != embedding_space:
                            raise ValueError("Provider embedding identity differs.")

                    new_vectors = await embed_batch(
                        provider,
                        batch_new_texts,
                    )

                    if len(new_vectors) != len(batch_new_texts):
                        raise ValueError("Embedding count mismatch.")

                    for vector in new_vectors:
                        validate_vector(vector)

                    reusable.update(
                        zip(
                            batch_new_texts,
                            new_vectors,
                            strict=True,
                        )
                    )
                    embedded_texts += len(batch_new_texts)

                vectors = [reusable[record["content"]] for record in batch]

                for vector in vectors:
                    validate_vector(vector)

                # Chunks and document progress commit atomically.
                async with sessions() as session:
                    async with session.begin():
                        for record, vector in zip(batch, vectors, strict=True):
                            session.add(
                                Chunk(
                                    id=UUID(record["chunk_id"]),
                                    document_id=document_id,
                                    chunk_index=record["chunk_index"],
                                    content=record["content"],
                                    embedding=vector,
                                    embedding_model=embedding_space,
                                    metadata_=record["metadata"],
                                )
                            )

                        await update_document_progress(
                            session,
                            document_id,
                            len(document_records),
                        )

                existing.update(UUID(record["chunk_id"]) for record in batch)
                copied_chunks += reused_in_batch
                batches += 1

                print(
                    f"Committed batch {batches}: "
                    f"{len(batch)} chunks; "
                    f"reused={reused_in_batch}; "
                    f"new_unique_texts={len(batch_new_texts)}; "
                    f"document={document_id}",
                    flush=True,
                )

            # Also repair progress for documents completed on an
            # earlier run, including runs stopped at a batch limit.
            async with sessions() as session:
                async with session.begin():
                    await update_document_progress(
                        session,
                        document_id,
                        len(document_records),
                    )

        remaining = len(expected) - len(existing)

        print(f"Copied existing embeddings: {copied_chunks}")
        print(f"New texts embedded: {embedded_texts}")
        print(f"Remaining chunks: {remaining}")

        if remaining:
            print("Reuse pass finished. Unmatched chunks still need embeddings.")
        else:
            print("Variant complete. All documents marked completed.")

    finally:
        try:
            if provider is not None:
                await provider.aclose()
        finally:
            await target_engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        type=int,
        choices=EXPERIMENT_SIZES,
        required=True,
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("data/experiments/token-chunks-v1"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reuse-only", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.batch_size <= 100:
        parser.error("Batch size must be between 1 and 100.")

    if args.max_batches is not None and args.max_batches < 1:
        parser.error("Max batches must be positive.")

    if args.reuse_only and not args.execute:
        parser.error("--reuse-only requires --execute.")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
