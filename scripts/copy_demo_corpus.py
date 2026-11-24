import argparse
import asyncio
import hashlib
import json
import math
import ssl
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from dotenv import dotenv_values
from sqlalchemy import func, insert, select, text
from sqlalchemy.ext.asyncio import create_async_engine

from docurag.core.config import Settings
from docurag.db.models import Chunk, Document

DOCUMENTS = Document.__table__
CHUNKS = Chunk.__table__

DOCUMENT_COLUMNS = list(DOCUMENTS.columns)
CHUNK_COLUMNS = [column for column in CHUNKS.columns if column.computed is None]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def encode_special(value):
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    raise TypeError(f"Unsupported value: {type(value).__name__}")


def fingerprint(documents: list[dict], chunks: list[dict]) -> str:
    payload = json.dumps(
        {"documents": documents, "chunks": chunks},
        sort_keys=True,
        separators=(",", ":"),
        default=encode_special,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


async def fetch_corpus(connection) -> tuple[list[dict], list[dict]]:
    document_rows = await connection.execute(
        select(*DOCUMENT_COLUMNS)
        .where(DOCUMENTS.c.status == "completed")
        .order_by(DOCUMENTS.c.id)
    )
    documents = [dict(row) for row in document_rows.mappings()]

    chunk_rows = await connection.execute(
        select(*CHUNK_COLUMNS)
        .join(DOCUMENTS, DOCUMENTS.c.id == CHUNKS.c.document_id)
        .where(DOCUMENTS.c.status == "completed")
        .order_by(CHUNKS.c.id)
    )
    chunks = [dict(row) for row in chunk_rows.mappings()]

    for chunk in chunks:
        vector = [float(value) for value in chunk["embedding"]]
        require(
            len(vector) == 768
            and all(math.isfinite(value) for value in vector)
            and any(value != 0 for value in vector),
            "Invalid embedding.",
        )
        require(
            chunk["embedding_model"] == "gemini:gemini-embedding-001:768",
            "Unexpected embedding space.",
        )
        chunk["embedding"] = vector

    return documents, chunks


async def require_empty(connection) -> None:
    for table in (DOCUMENTS, CHUNKS):
        count = await connection.scalar(select(func.count()).select_from(table))
        require(count == 0, f"Destination {table.name} is not empty.")


async def run(args) -> None:
    source = Settings(_env_file=None, db_ssl_mode="disable")

    values = dotenv_values(args.target_env)
    required = (
        "DB_HOST",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "PGSSLROOTCERT",
    )
    require(
        all(values.get(key) for key in required),
        "Target configuration is incomplete.",
    )
    require(
        values.get("DB_SSL_MODE") == "verify-full",
        "Target must use verify-full.",
    )

    target = Settings(
        _env_file=None,
        db_host=values["DB_HOST"],
        db_port=int(values.get("DB_PORT") or 5432),
        postgres_db=values["POSTGRES_DB"],
        postgres_user=values["POSTGRES_USER"],
        postgres_password=values["POSTGRES_PASSWORD"],
        db_ssl_mode="verify-full",
    )
    require(
        target.db_host.endswith(".neon.tech") and "-pooler" not in target.db_host,
        "Target must be a direct Neon endpoint.",
    )
    require(
        source.db_host != target.db_host,
        "Source and destination hosts must differ.",
    )

    tls = ssl.create_default_context(cafile=values["PGSSLROOTCERT"])

    source_engine = create_async_engine(
        source.database_url,
        isolation_level="REPEATABLE READ",
    )
    target_engine = create_async_engine(
        target.database_url,
        connect_args={"ssl": tls, "timeout": 20},
    )

    try:
        async with source_engine.connect() as connection:
            async with connection.begin():
                await connection.execute(text("SET TRANSACTION READ ONLY"))
                documents, chunks = await fetch_corpus(connection)

        require(len(documents) == 38, "Expected 38 completed documents.")
        require(len(chunks) == 991, "Expected 991 baseline chunks.")

        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        expected = {item["document_id"]: item for item in manifest}
        require(
            {str(document["id"]) for document in documents} == set(expected),
            "Source documents differ from the frozen inventory.",
        )

        counts = Counter(chunk["document_id"] for chunk in chunks)

        for document in documents:
            original = expected[str(document["id"])]
            require(
                document["filename"] == original["filename"]
                and document["content_hash"] == original["content_hash"],
                "Source document identity changed.",
            )
            require(
                counts[document["id"]] == document["chunk_count"],
                "Source document chunk count is inconsistent.",
            )
            document["storage_path"] = None

        expected_fingerprint = fingerprint(documents, chunks)

        async with target_engine.connect() as connection:
            version = await connection.scalar(
                text("SELECT version_num FROM alembic_version")
            )
            require(version == "0003", "Expected destination migration 0003.")
            await require_empty(connection)

        print(f"Source: {len(documents)} documents; {len(chunks)} chunks.")
        print("Destination: empty Neon tables at migration 0003.")
        print(f"Transfer fingerprint: {expected_fingerprint}")

        if not args.execute:
            print("DRY RUN: no database writes or model API calls.")
            return

        async with target_engine.begin() as connection:
            await connection.execute(text("SET LOCAL lock_timeout = '5s'"))
            await connection.execute(
                text("LOCK TABLE documents, chunks IN SHARE ROW EXCLUSIVE MODE")
            )
            await require_empty(connection)

            await connection.execute(insert(DOCUMENTS), documents)

            for start in range(0, len(chunks), 100):
                await connection.execute(
                    insert(CHUNKS),
                    chunks[start : start + 100],
                )

            copied_documents, copied_chunks = await fetch_corpus(connection)
            require(
                fingerprint(copied_documents, copied_chunks) == expected_fingerprint,
                "Destination verification failed; transaction rolled back.",
            )

        print("PASS: committed and verified 38 documents and 991 chunks.")
        print("Existing embeddings reused. No model API calls.")

    finally:
        await source_engine.dispose()
        await target_engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-env",
        type=Path,
        default=Path("/app/.env.render-demo"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/experiments/source-manifest.json"),
    )
    parser.add_argument("--execute", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
