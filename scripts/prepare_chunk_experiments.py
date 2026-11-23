import argparse
import hashlib
import inspect
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from uuid import NAMESPACE_URL, uuid5

from docurag.core.config import Settings
from docurag.ingestion import chunkers, parsers
from docurag.ingestion.chunkers import chunk_sections
from docurag.ingestion.parsers import (
    SUPPORTED_EXTENSIONS,
    parse_document,
)

SIZES = (256, 512, 1024)
OVERLAP = 50


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def normalize(value: str) -> str:
    return " ".join(value.split())


def save(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/experiments/source-manifest.json"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/eval/v1/dev.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/experiments/token-chunks-v1"),
    )
    args = parser.parse_args()

    if args.output.exists():
        raise ValueError(
            "Output directory already exists. "
            "Use a new path to preserve the previous preparation."
        )

    if args.dataset.stem != "dev":
        raise ValueError("Evidence checks are restricted to dev.json.")

    manifest_raw = args.manifest.read_bytes()
    documents = json.loads(manifest_raw)

    if not documents:
        raise ValueError("Completed-document manifest is empty.")

    filenames = [item["filename"] for item in documents]

    if len(set(filenames)) != len(filenames):
        raise ValueError("Duplicate filenames need explicit path mapping.")

    dataset_raw = args.dataset.read_bytes()
    questions = json.loads(dataset_raw)
    frozen = json.loads((args.dataset.parent / "manifest.json").read_text())

    if frozen["dev_sha256"] != digest(dataset_raw):
        raise ValueError("Development dataset differs from its manifest.")

    available = defaultdict(list)

    for root in (Path("data/corpus"), Path("data/samples")):
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                available[path.name].append(path)

    sections_by_document = {}
    verified_sources = []

    for document in documents:
        filename = document["filename"]
        matches = available[filename]

        if len(matches) != 1:
            raise ValueError(
                f"{filename}: expected one local file, found {len(matches)}."
            )

        path = matches[0]
        file_hash = digest(path.read_bytes())

        if file_hash != document["content_hash"]:
            raise ValueError(
                f"{filename}: local bytes differ from the ingested source."
            )

        sections = parse_document(path, source_file=filename)
        sections_by_document[document["document_id"]] = sections
        verified_sources.append(
            {
                **document,
                "local_path": str(path),
                "section_count": len(sections),
            }
        )

    document_ids = set(sections_by_document)

    if any(q["document_id"] not in document_ids for q in questions):
        raise ValueError("A development question's source is missing.")

    unused = sorted(set(available) - set(filenames))

    # Explicit placeholder: chunk preparation does not connect to Postgres.
    settings = Settings(postgres_password="offline-unused")

    args.output.mkdir(parents=True)
    save(args.output / "sources.json", verified_sources)

    identity = {
        "source_manifest_sha256": digest(manifest_raw),
        "dev_dataset_sha256": digest(dataset_raw),
        "parser_sha256": digest(inspect.getsource(parsers).encode()),
        "chunker_sha256": digest(inspect.getsource(chunkers).encode()),
        "sizes_tokens": list(SIZES),
        "overlap_tokens": OVERLAP,
        "unused_local_files": unused,
    }
    save(args.output / "manifest.json", identity)

    print(f"Verified completed documents: {len(documents)}")
    print(f"Unused local files: {unused}")
    print("\nSize | Chunks | Mean tokens | Max tokens | Evidence covered")

    summary = []

    for size in SIZES:
        variant_settings = settings.model_copy(
            update={
                "chunker": "token",
                "token_chunk_size": size,
                "token_chunk_overlap": OVERLAP,
            }
        )

        records = []
        by_document = defaultdict(list)

        for document in documents:
            document_id = document["document_id"]
            chunks = chunk_sections(
                sections_by_document[document_id],
                variant_settings,
            )

            for chunk in chunks:
                if chunk.metadata["token_count"] > size:
                    raise ValueError("Chunk exceeds configured token limit.")

                content_hash = digest(chunk.content.encode())
                chunk_id = str(
                    uuid5(
                        NAMESPACE_URL,
                        (
                            f"docurag:token:{size}:{OVERLAP}:"
                            f"{document_id}:{chunk.chunk_index}:{content_hash}"
                        ),
                    )
                )

                record = {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                }
                records.append(record)
                by_document[document_id].append(record)

        evidence_checks = []

        for question in questions:
            evidence = normalize(question["evidence_text"])

            if not evidence:
                raise ValueError("Question has empty evidence.")

            matches = [
                chunk["chunk_id"]
                for chunk in by_document[question["document_id"]]
                if evidence in normalize(chunk["content"])
            ]

            evidence_checks.append(
                {
                    "question_id": question["id"],
                    "full_quote_in_one_chunk": bool(matches),
                    "matching_chunk_ids": matches,
                }
            )

        counts = [item["metadata"]["token_count"] for item in records]
        covered = sum(item["full_quote_in_one_chunk"] for item in evidence_checks)

        row = {
            "chunk_size_tokens": size,
            "overlap_tokens": OVERLAP,
            "documents": len(documents),
            "chunks": len(records),
            "mean_tokens": round(mean(counts), 2),
            "max_tokens": max(counts),
            "dev_evidence_covered": covered,
            "dev_questions": len(questions),
        }
        summary.append(row)

        save(args.output / f"token-{size}.json", records)
        save(
            args.output / f"token-{size}-evidence.json",
            evidence_checks,
        )

        print(
            f"{size} | {len(records)} | {row['mean_tokens']} "
            f"| {max(counts)} | {covered}/{len(questions)}"
        )

    save(args.output / "summary.json", summary)
    print(f"\nSaved: {args.output}")
    print("No embeddings, generation calls or database writes performed.")


if __name__ == "__main__":
    main()
