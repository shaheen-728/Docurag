import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
)

from docurag.core.config import get_settings
from docurag.db.models import Chunk, Document


async def run(args: argparse.Namespace) -> None:
    settings = get_settings()
    target_database = f"docurag_eval_token{args.size}"

    if settings.postgres_db == target_database:
        raise ValueError("Run with the baseline database configured as POSTGRES_DB.")

    sources_raw = (args.prepared / "sources.json").read_bytes()
    sources = json.loads(sources_raw)
    manifest_raw = (args.prepared / "manifest.json").read_bytes()

    preparation_hash = hashlib.sha256(manifest_raw + sources_raw).hexdigest()

    source_engine = create_async_engine(settings.database_url)
    target_settings = settings.model_copy(update={"postgres_db": target_database})
    target_engine = create_async_engine(target_settings.database_url)

    source_sessions = async_sessionmaker(
        source_engine,
        expire_on_commit=False,
    )
    target_sessions = async_sessionmaker(
        target_engine,
        expire_on_commit=False,
    )

    try:
        # Read baseline records without modifying them.
        async with source_sessions() as session:
            records = []

            for source in sources:
                document = await session.get(
                    Document,
                    UUID(source["document_id"]),
                )

                if document is None or document.status != "completed":
                    raise ValueError("Baseline document is missing/incomplete.")

                if (
                    document.filename != source["filename"]
                    or document.content_hash != source["content_hash"]
                ):
                    raise ValueError(f"Baseline source changed: {source['filename']}")

                metadata = dict(document.metadata_)
                metadata["_chunk_experiment"] = {
                    "size_tokens": args.size,
                    "overlap_tokens": 50,
                    "preparation_sha256": preparation_hash,
                }

                records.append(
                    {
                        "id": document.id,
                        "filename": document.filename,
                        "content_type": document.content_type,
                        "content_hash": document.content_hash,
                        "created_at": document.created_at,
                        "metadata_": metadata,
                    }
                )

        expected_ids = {record["id"] for record in records}

        if not expected_ids or len(expected_ids) != len(records):
            raise ValueError("Expected unique, nonempty source records.")

        # One transaction: either all document records are seeded or none.
        async with target_sessions() as session:
            async with session.begin():
                existing = (await session.scalars(select(Document))).all()

                if any(doc.id not in expected_ids for doc in existing):
                    raise ValueError(
                        "Target contains documents outside this experiment."
                    )

                by_id = {doc.id: doc for doc in existing}
                inserted = 0

                for record in records:
                    current = by_id.get(record["id"])

                    if current is not None:
                        if (
                            current.filename != record["filename"]
                            or current.content_hash != record["content_hash"]
                            or current.metadata_.get("_chunk_experiment")
                            != record["metadata_"]["_chunk_experiment"]
                        ):
                            raise ValueError(
                                "Existing target document has a "
                                "different experiment identity."
                            )
                        continue

                    session.add(
                        Document(
                            **record,
                            status="queued",
                            chunk_count=0,
                            storage_path=None,
                        )
                    )
                    inserted += 1

        async with target_sessions() as session:
            document_count = await session.scalar(
                select(func.count()).select_from(Document)
            )
            chunk_count = await session.scalar(select(func.count()).select_from(Chunk))

        print(f"Target: {target_database}")
        print(f"New documents: {inserted}")
        print(f"Total documents: {document_count}")
        print(f"Existing chunks: {chunk_count}")
        print("Baseline unchanged. No model API calls made.")

    finally:
        await source_engine.dispose()
        await target_engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        type=int,
        choices=(256, 512, 1024),
        required=True,
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("data/experiments/token-chunks-v1"),
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
