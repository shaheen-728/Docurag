import argparse
import hashlib
import json
import time
from pathlib import Path

import httpx

MODES = {
    "keyword": {"mode": "keyword", "rerank": False},
    "vector": {"mode": "vector", "rerank": False},
    "hybrid": {"mode": "hybrid", "rerank": False},
    "hybrid_rerank": {"mode": "hybrid", "rerank": True},
}


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def collect(args: argparse.Namespace) -> None:
    raw = args.dataset.read_bytes()
    questions = json.loads(raw)

    if len({question["id"] for question in questions}) != len(questions):
        raise ValueError("Dataset contains duplicate question IDs.")

    manifest_path = args.dataset.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_hash = hashlib.sha256(raw).hexdigest()
    expected_hash = manifest.get(f"{args.dataset.stem}_sha256")

    if expected_hash != dataset_hash:
        raise ValueError("Dataset does not match the frozen manifest.")

    if args.dataset.stem not in {"dev", "test"}:
        raise ValueError("Only frozen dev.json or test.json is supported.")

    selected_modes = {name: MODES[name] for name in args.modes}

    if args.dataset.stem == "test":
        if not args.allow_heldout:
            raise ValueError("Held-out collection requires --allow-heldout.")

        if len(questions) != 20 or args.limit != 20:
            raise ValueError("Held-out collection must include all 20 questions.")

        if set(selected_modes) != {"vector"}:
            raise ValueError("The frozen held-out mode is vector only.")

        heldout_directory = Path("results/day5/heldout").resolve()

        if args.output.resolve().parent != heldout_directory:
            raise ValueError("Save held-out results under results/day5/heldout.")

        dev_raw = (args.dataset.parent / "dev.json").read_bytes()

        if hashlib.sha256(dev_raw).hexdigest() != manifest.get("dev_sha256"):
            raise ValueError("Development dataset differs from its manifest.")

        dev_ids = {item["id"] for item in json.loads(dev_raw)}
        test_ids = {item["id"] for item in questions}

        if dev_ids & test_ids:
            raise ValueError("Development and test question IDs overlap.")

    identity = {
        "dataset_sha256": dataset_hash,
        "base_url": args.base_url.rstrip("/"),
        "corpus_label": args.corpus_label,
        "top_k": 5,
        "candidate_k": 20,
        "modes": selected_modes,
    }

    if args.output.exists():
        run = json.loads(args.output.read_text(encoding="utf-8"))

        if run["identity"] != identity:
            raise ValueError("Run configuration changed. Use a new output path.")
    else:
        run = {
            "identity": identity,
            "dataset_path": str(args.dataset),
            "label_provenance": "assistant-reviewed",
            "relevance_scope": "generation_source_only",
            "responses": {},
            "errors": [],
        }
        save_json(args.output, run)

    selected = questions[: args.limit]
    total_requests = len(selected) * len(selected_modes)

    print(
        f"Selected questions: {len(selected)}; method requests: {total_requests}",
        flush=True,
    )

    with httpx.Client(timeout=300) as client:
        for question in selected:
            question_id = question["id"]
            responses = run["responses"].setdefault(
                question_id,
                {},
            )

            for name, options in selected_modes.items():
                if name in responses:
                    print(f"{question_id} {name}: reused")
                    continue

                print(
                    f"{question_id} {name}: requesting",
                    flush=True,
                )

                try:
                    started = time.perf_counter()
                    response = client.post(
                        f"{identity['base_url']}/search",
                        json={
                            "query": question["question"],
                            "top_k": identity["top_k"],
                            "candidate_k": identity["candidate_k"],
                            "use_cache": False,
                            **options,
                        },
                    )
                    response.raise_for_status()
                    result = response.json()

                    if result["mode"] != options["mode"]:
                        raise ValueError("Unexpected retrieval mode.")

                    hits = result["results"]

                    if (
                        name == "hybrid_rerank"
                        and hits
                        and not result.get("rerank_applied")
                    ):
                        raise ValueError("Reranking was not applied.")

                    chunk_ids = [hit["chunk_id"] for hit in hits]

                    if len(chunk_ids) != len(set(chunk_ids)):
                        raise ValueError("Duplicate returned chunk IDs.")

                    if len(hits) > identity["top_k"]:
                        raise ValueError("Too many returned results.")

                    responses[name] = {
                        "client_latency_ms": round(
                            (time.perf_counter() - started) * 1000,
                            3,
                        ),
                        "response": result,
                    }
                    save_json(args.output, run)

                    print(
                        f"  results={len(hits)} "
                        f"server_latency_ms={result['latency_ms']}",
                        flush=True,
                    )

                except Exception as error:
                    run["errors"].append(
                        {
                            "question_id": question_id,
                            "mode": name,
                            "error_type": type(error).__name__,
                        }
                    )
                    save_json(args.output, run)
                    raise

                time.sleep(args.pause)

    print(f"Saved: {args.output}")
    print("Keep the corpus and retrieval settings unchanged when resuming this run.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["collect"])
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/eval/v1/dev.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/day5/baseline/run.json"),
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--corpus-label",
        default="baseline-recursive-1200-overlap-200",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(MODES),
        default=list(MODES),
    )
    parser.add_argument("--allow-heldout", action="store_true")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--pause", type=float, default=4)
    args = parser.parse_args()

    if args.limit < 1 or args.pause < 0:
        parser.error("Limit must be positive; pause cannot be negative.")

    collect(args)


if __name__ == "__main__":
    main()
