import argparse
import asyncio
import hashlib
import inspect
import json
from contextlib import AsyncExitStack
from math import isfinite
from pathlib import Path
from time import perf_counter
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
)

from docurag.core.config import get_settings
from docurag.db.models import Chunk, Document
from docurag.retrieval import service
from docurag.schemas.search import SearchRequest

SIZES = (256, 512, 1024)


def checksum(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encoded(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_vector(vector: list, dimensions: int) -> None:
    if (
        len(vector) != dimensions
        or any(type(value) not in {int, float} for value in vector)
        or not all(isfinite(value) for value in vector)
        or not any(value != 0 for value in vector)
    ):
        raise ValueError("Invalid query embedding.")


async def inspect_corpus(
    session,
    sources: list[dict],
    embedding_space: str,
    prepared_records: list[dict] | None,
    preparation_hash: str,
    size: int | None,
) -> dict:
    source_by_id = {UUID(item["document_id"]): item for item in sources}

    document_query = select(Document).order_by(Document.id)

    if size is None:
        # Match retrieval behavior: exclude incomplete baseline uploads.
        document_query = document_query.where(Document.status == "completed")

    documents = (await session.scalars(document_query)).all()

    if {doc.id for doc in documents} != set(source_by_id):
        raise ValueError("Database documents differ from source inventory.")

    chunk_query = select(Chunk).order_by(Chunk.id)

    if size is None:
        chunk_query = chunk_query.where(Chunk.document_id.in_(list(source_by_id)))

    chunks = (await session.scalars(chunk_query)).all()

    for document in documents:
        source = source_by_id[document.id]

        if (
            document.status != "completed"
            or document.filename != source["filename"]
            or document.content_hash != source["content_hash"]
        ):
            raise ValueError(f"Incomplete or changed document: {document.filename}")

        if size is not None:
            marker = document.metadata_.get("_chunk_experiment", {})

            if (
                marker.get("size_tokens") != size
                or marker.get("overlap_tokens") != 50
                or marker.get("preparation_sha256") != preparation_hash
            ):
                raise ValueError("Experiment preparation identity differs.")

    expected = (
        {UUID(record["chunk_id"]): record for record in prepared_records}
        if prepared_records is not None
        else None
    )

    if expected is not None and {c.id for c in chunks} != set(expected):
        raise ValueError("Stored chunk IDs differ from prepared chunks.")

    if not chunks:
        raise ValueError("Database has no chunks.")

    counts = {}
    fingerprint = hashlib.sha256()

    for document in documents:
        fingerprint.update(
            encoded(
                {
                    "id": str(document.id),
                    "filename": document.filename,
                    "content_hash": document.content_hash,
                    "status": document.status,
                    "chunk_count": document.chunk_count,
                    "metadata": document.metadata_,
                }
            )
        )

    for chunk in chunks:
        if chunk.embedding_model != embedding_space:
            raise ValueError("Database embedding space differs.")

        if chunk.document_id not in source_by_id:
            raise ValueError("Chunk belongs to an unexpected document.")

        if expected is not None:
            record = expected[chunk.id]

            if (
                str(chunk.document_id) != record["document_id"]
                or chunk.chunk_index != record["chunk_index"]
                or chunk.content != record["content"]
                or chunk.metadata_ != record["metadata"]
            ):
                raise ValueError("Stored chunk differs from preparation.")

        vector = [float(value) for value in chunk.embedding]
        validate_vector(vector, 768)

        counts[chunk.document_id] = counts.get(chunk.document_id, 0) + 1

        fingerprint.update(
            encoded(
                {
                    "id": str(chunk.id),
                    "document_id": str(chunk.document_id),
                    "index": chunk.chunk_index,
                    "content": chunk.content,
                    "metadata": chunk.metadata_,
                    "embedding_model": chunk.embedding_model,
                    "embedding": vector,
                }
            )
        )

    for document in documents:
        if counts.get(document.id, 0) != document.chunk_count:
            raise ValueError("Document chunk count differs from stored rows.")

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "corpus_sha256": fingerprint.hexdigest(),
    }


async def run(args: argparse.Namespace) -> None:
    settings = get_settings()

    if settings.postgres_db.startswith("docurag_eval_"):
        raise ValueError("Configure POSTGRES_DB as the baseline database.")

    if args.dataset.name != "dev.json":
        raise ValueError("Only the development dataset is allowed.")

    dataset_raw = args.dataset.read_bytes()
    dataset_hash = checksum(dataset_raw)
    questions = json.loads(dataset_raw)

    frozen = json.loads(
        (args.dataset.parent / "manifest.json").read_text(encoding="utf-8")
    )

    if frozen["dev_sha256"] != dataset_hash:
        raise ValueError("Dataset differs from the frozen manifest.")

    if not questions or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Dataset must contain unique question IDs.")

    manifest_raw = (args.prepared / "manifest.json").read_bytes()
    source_raw = (args.prepared / "sources.json").read_bytes()
    preparation = json.loads(manifest_raw)
    sources = json.loads(source_raw)

    if preparation["dev_dataset_sha256"] != dataset_hash:
        raise ValueError("Preparation belongs to another dataset.")

    preparation_hash = checksum(manifest_raw + source_raw)

    cache_raw = args.query_cache.read_bytes()
    cache = json.loads(cache_raw)

    model = getattr(
        settings,
        f"{settings.llm_provider}_embedding_model",
    )
    embedding_space = f"{settings.llm_provider}:{model}:{settings.embedding_dimensions}"

    if cache["identity"] != {
        "schema_version": 1,
        "dataset_sha256": dataset_hash,
        "embedding_space": embedding_space,
        "task": "query",
    }:
        raise ValueError("Query cache identity differs.")

    for question in questions:
        key = checksum(question["question"].encode("utf-8"))

        if key not in cache["vectors"]:
            raise ValueError(f"Missing query embedding: {question['id']}")

        validate_vector(
            cache["vectors"][key],
            settings.embedding_dimensions,
        )

    variants = {
        "baseline": (settings.postgres_db, None),
        **{f"token{size}": (f"docurag_eval_token{size}", size) for size in SIZES},
    }

    engines = []
    sessions = {}
    inventories = {}
    prepared_hashes = {}

    try:
        async with AsyncExitStack() as stack:
            for name, (database, size) in variants.items():
                target_settings = settings.model_copy(update={"postgres_db": database})
                engine = create_async_engine(
                    target_settings.database_url,
                    isolation_level="REPEATABLE READ",
                )
                engines.append(engine)

                factory = async_sessionmaker(engine)
                session = await stack.enter_async_context(factory())

                # Keep a consistent, read-only corpus snapshot.
                await session.execute(text("SET TRANSACTION READ ONLY"))
                sessions[name] = session

                records = None

                if size is not None:
                    raw = (args.prepared / f"token-{size}.json").read_bytes()
                    prepared_hashes[name] = checksum(raw)
                    records = json.loads(raw)

                inventories[name] = await inspect_corpus(
                    session,
                    sources,
                    embedding_space,
                    records,
                    preparation_hash,
                    size,
                )

                inventory = inventories[name]
                print(
                    f"{name}: documents={inventory['documents']} "
                    f"chunks={inventory['chunks']}",
                    flush=True,
                )

            identity = {
                "schema_version": 1,
                "dataset_sha256": dataset_hash,
                "query_cache_sha256": checksum(cache_raw),
                "preparation_sha256": preparation_hash,
                "prepared_chunk_sha256": prepared_hashes,
                "embedding_space": embedding_space,
                "retrieval_service_sha256": checksum(
                    inspect.getsource(service).encode("utf-8")
                ),
                "top_k": 5,
                "candidate_k": 20,
                "mode": "vector",
                "rerank": False,
                "variants": {
                    name: {
                        "database": database,
                        **inventories[name],
                    }
                    for name, (database, _) in variants.items()
                },
            }

            if args.output.exists():
                state = json.loads(args.output.read_text(encoding="utf-8"))

                if state["identity"] != identity:
                    raise ValueError("Run identity changed. Use a new output path.")
            else:
                state = {
                    "identity": identity,
                    "dataset_path": str(args.dataset),
                    "latency_scope": "database_vector_retrieval_only",
                    "label_status": "awaiting_variant_relevance_review",
                    "responses": {},
                }

            selected = questions[: args.limit]

            print(f"Selected questions: {len(selected)}")
            print("Model API calls required: 0")

            if not args.execute:
                print("DRY RUN: no result files or database writes.")
                return

            save(args.output, state)

            for question in selected:
                question_id = question["id"]
                key = checksum(question["question"].encode("utf-8"))
                vector = cache["vectors"][key]
                responses = state["responses"].setdefault(question_id, {})

                request = SearchRequest(
                    query=question["question"],
                    mode="vector",
                    top_k=5,
                    candidate_k=20,
                    rerank=False,
                    use_cache=False,
                )

                for name in variants:
                    if name in responses:
                        print(f"{question_id} {name}: reused")
                        continue

                    started = perf_counter()
                    candidates = await service.vector_search(
                        sessions[name],
                        request,
                        vector,
                        embedding_space,
                    )
                    latency_ms = round(
                        (perf_counter() - started) * 1000,
                        3,
                    )

                    results = candidates[: request.top_k]

                    responses[name] = {
                        "retrieval_latency_ms": latency_ms,
                        "candidate_count": len(candidates),
                        "results": [hit.model_dump(mode="json") for hit in results],
                    }
                    save(args.output, state)

                    print(
                        f"{question_id} {name}: "
                        f"results={len(results)} "
                        f"retrieval_ms={latency_ms}",
                        flush=True,
                    )

            print(f"Saved: {args.output}")
            print("Relevance review is required before scoring.")

    finally:
        for engine in engines:
            await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/eval/v1/dev.json"),
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("data/experiments/token-chunks-v1"),
    )
    parser.add_argument(
        "--query-cache",
        type=Path,
        default=Path("data/experiments/query-embeddings-dev-v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/day5/chunk-sizes/run.json"),
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("Limit must be positive.")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
