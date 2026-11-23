import argparse
import asyncio
import hashlib
import json
from math import isfinite
from pathlib import Path

from docurag.core.config import get_settings
from docurag.llm.factory import build_provider


def checksum(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def query_key(query: str) -> str:
    return checksum(query.encode("utf-8"))


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
        raise ValueError("Invalid cached query embedding.")


async def run(args: argparse.Namespace) -> None:
    settings = get_settings()

    if args.dataset.name != "dev.json":
        raise ValueError("Use the development dataset only.")

    dataset_raw = args.dataset.read_bytes()
    dataset_hash = checksum(dataset_raw)
    manifest = json.loads(
        (args.dataset.parent / "manifest.json").read_text(encoding="utf-8")
    )

    if manifest["dev_sha256"] != dataset_hash:
        raise ValueError("Dataset differs from the frozen manifest.")

    questions = json.loads(dataset_raw)

    if not questions:
        raise ValueError("Development dataset is empty.")

    if len({item["id"] for item in questions}) != len(questions):
        raise ValueError("Duplicate question IDs.")

    model = getattr(
        settings,
        f"{settings.llm_provider}_embedding_model",
    )
    embedding_space = f"{settings.llm_provider}:{model}:{settings.embedding_dimensions}"

    identity = {
        "schema_version": 1,
        "dataset_sha256": dataset_hash,
        "embedding_space": embedding_space,
        "task": "query",
    }

    queries = {}

    for item in questions:
        query = item["question"]

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Invalid development question.")

        queries[query_key(query)] = query

    if args.output.exists():
        state = json.loads(args.output.read_text(encoding="utf-8"))

        if state["identity"] != identity:
            raise ValueError("Query cache identity changed. Use a new output file.")
    else:
        state = {
            "identity": identity,
            "vectors": {},
        }

    for key, vector in state["vectors"].items():
        if key not in queries:
            raise ValueError("Cache contains an unknown query.")

        validate_vector(vector, settings.embedding_dimensions)

    missing = [
        (key, query) for key, query in queries.items() if key not in state["vectors"]
    ]

    print(f"Development questions: {len(questions)}")
    print(f"Unique query texts: {len(queries)}")
    print(f"Cached: {len(state['vectors'])}")
    print(f"Missing: {len(missing)}")
    print(f"Embedding space: {embedding_space}")

    if not args.execute:
        print("DRY RUN: no API calls or file writes.")
        return

    if not missing:
        print("Query cache is complete. No API calls needed.")
        return

    save(args.output, state)
    provider = build_provider(settings)

    try:
        if provider.embedding_space != embedding_space:
            raise ValueError("Provider embedding identity differs.")

        selected = missing[: args.max_calls]

        for index, (key, query) in enumerate(selected, start=1):
            print(
                f"Embedding query {index}/{len(selected)}",
                flush=True,
            )

            async with asyncio.timeout(60):
                vectors = await provider.embed(
                    [query],
                    task="query",
                )

            if len(vectors) != 1:
                raise ValueError("Expected one query embedding.")

            validate_vector(
                vectors[0],
                settings.embedding_dimensions,
            )

            state["vectors"][key] = vectors[0]
            save(args.output, state)

            print(
                f"Saved query embeddings: {len(state['vectors'])}",
                flush=True,
            )

            if index < len(selected):
                await asyncio.sleep(args.pause)

    finally:
        await provider.aclose()

    print(f"Saved: {args.output}")
    print(f"Remaining: {len(queries) - len(state['vectors'])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/eval/v1/dev.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/query-embeddings-dev-v1.json"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-calls", type=int, default=5)
    parser.add_argument("--pause", type=float, default=4)
    args = parser.parse_args()

    if args.max_calls < 1 or args.pause < 0:
        parser.error("Max calls must be positive; pause cannot be negative.")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
