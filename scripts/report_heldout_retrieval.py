"""Report frozen held-out vector retrieval against reviewed pooled labels."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> tuple[dict | list, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def evaluate(run: dict, review: dict, dataset: list) -> dict:
    identity = run["identity"]
    require(identity["top_k"] == 5, "Expected top_k=5.")
    require(identity["candidate_k"] == 20, "Expected candidate_k=20.")
    require(
        identity["modes"] == {"vector": {"mode": "vector", "rerank": False}},
        "Expected the frozen vector-only configuration.",
    )
    require(
        review["label_scope"] == "pooled_top5_plus_generation_source",
        "Unexpected review pool scope.",
    )
    questions = {q["id"]: q for q in dataset}
    reviews = {q["id"]: q for q in review["questions"]}
    require(len(questions) == len(dataset) == 20, "Expected 20 unique questions.")
    require(
        len(reviews) == len(review["questions"]) == 20,
        "Expected 20 unique reviewed questions.",
    )
    require(
        set(questions) == set(reviews) == set(run["responses"]),
        "Run, dataset and review question sets differ.",
    )
    rows = []
    for qid, question in questions.items():
        assessed = reviews[qid]
        require(
            assessed["reviewed"] is True and bool(assessed["reviewer"]),
            f"{qid}: review incomplete.",
        )
        for key in ("question", "reference_answer", "evidence_text"):
            require(assessed[key] == question[key], f"{qid}: changed {key}.")
        candidates = {c["chunk_id"]: c for c in assessed["candidates"]}
        require(
            len(candidates) == len(assessed["candidates"]),
            f"{qid}: duplicate review candidates.",
        )
        for candidate in candidates.values():
            require(
                type(candidate["relevant"]) is bool, f"{qid}: missing relevance label."
            )
            require(bool(candidate["review_notes"]), f"{qid}: missing notes.")
            require(
                candidate["variants"] == ["vector"],
                f"{qid}: unexpected candidate variants.",
            )
        saved = run["responses"][qid]
        require(set(saved) == {"vector"}, f"{qid}: unexpected retrieval modes.")
        saved = saved["vector"]
        response = saved["response"]
        require(response["mode"] == "vector", f"{qid}: incorrect mode.")
        require(not response.get("rerank_applied", False), f"{qid}: reranked.")
        require(not response.get("cache_hit", False), f"{qid}: cached result.")
        hits = response["results"]
        retrieved = [h["chunk_id"] for h in hits]
        require(
            len(retrieved) == len(set(retrieved)) == 5,
            f"{qid}: expected five unique results.",
        )
        require(set(retrieved) <= set(candidates), f"{qid}: unreviewed result.")
        for hit in hits:
            candidate = candidates[hit["chunk_id"]]
            require(
                all(
                    candidate[key] == hit[key]
                    for key in ("document_id", "filename", "content")
                ),
                f"{qid}: reviewed content differs from the result.",
            )
        relevant = {cid for cid, c in candidates.items() if c["relevant"]}
        require(bool(relevant), f"{qid}: no known positive in the pool.")
        source_id = question["relevant_chunk_ids"][0]
        require(source_id in relevant, f"{qid}: generation source missing.")
        source = candidates[source_id]
        require(
            source["content"] == question["source_content"]
            and source["document_id"] == question["document_id"],
            f"{qid}: generation-source content changed.",
        )
        found = set(retrieved) & relevant
        rank = next((i for i, cid in enumerate(retrieved, 1) if cid in relevant), None)
        server_ms = response["latency_ms"]
        client_ms = saved["client_latency_ms"]
        require(
            all(
                type(x) in (int, float) and math.isfinite(x) and x >= 0
                for x in (server_ms, client_ms)
            ),
            f"{qid}: invalid latency.",
        )
        rows.append(
            {
                "question_id": qid,
                "hit_at_5": float(bool(found)),
                "reciprocal_rank_at_5": 1.0 / rank if rank else 0.0,
                "pooled_recall_at_5": len(found) / len(relevant),
                "precision_at_5": len(found) / 5,
                "first_relevant_rank": rank,
                "pooled_relevant": len(relevant),
                "retrieved_relevant": len(found),
                "server_latency_ms": server_ms,
                "client_latency_ms": client_ms,
            }
        )
    metrics = {
        key: mean(row[key] for row in rows)
        for key in (
            "hit_at_5",
            "reciprocal_rank_at_5",
            "pooled_recall_at_5",
            "precision_at_5",
            "server_latency_ms",
            "client_latency_ms",
        )
    }
    return {
        "complete": True,
        "question_count": 20,
        "metrics": metrics,
        "per_question": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=Path("results/day5/heldout/run.json")
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=Path("results/day5/heldout/relevance_review.json"),
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/eval/v1/test.json"))
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/day5/heldout/retrieval-metrics"),
    )
    args = parser.parse_args()
    run, run_hash = read_json(args.run)
    review, review_hash = read_json(args.review)
    dataset, dataset_hash = read_json(args.dataset)
    require(review["run_sha256"] == run_hash, "Review run hash mismatch.")
    require(
        run["identity"]["dataset_sha256"] == dataset_hash
        and review["dataset_sha256"] == dataset_hash,
        "Dataset hash mismatch.",
    )
    report = evaluate(run, review, dataset)
    report.update(
        {
            "run_sha256": run_hash,
            "review_sha256": review_hash,
            "dataset_sha256": dataset_hash,
            "identity": run["identity"],
            "label_scope": review["label_scope"],
            "label_provenance": "assistant-reviewed; not human-verified",
        }
    )
    m = report["metrics"]
    hit_count = sum(row["hit_at_5"] for row in report["per_question"])
    lines = [
        "# Held-out retrieval evaluation",
        "",
        "Questions: 20. Baseline chunking. Vector retrieval. Top k: 5.",
        "",
        "| Mode | Hit@5 | MRR@5 | Pooled Recall@5 | Precision@5 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| vector | {m['hit_at_5']:.3f} | "
        f"{m['reciprocal_rank_at_5']:.3f} | "
        f"{m['pooled_recall_at_5']:.3f} | {m['precision_at_5']:.3f} |",
        "",
        f"Retrieval hits: {hit_count:.0f}/20.",
        f"Mean API-reported search latency: {m['server_latency_ms']:.2f} ms.",
        f"Mean client-observed search latency: {m['client_latency_ms']:.2f} ms.",
        "",
        "Labels are assistant-reviewed, not human-verified; partial support counts.",
        "Recall uses the top-5 plus generation-source pool, not exhaustive labels.",
        "This pool differs from the chunk-size comparison pool.",
        "Search latencies are not the database-only chunk-size timings.",
        "These are retrieval metrics, not generated-answer correctness scores.",
        "Do not tune the selected configuration using these held-out results.",
        "No model API calls are made by this reporting script.",
        "",
        f"Run SHA256: {run_hash}",
        f"Review SHA256: {review_hash}",
    ]
    markdown = "\n".join(lines) + "\n"
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.output_prefix) + ".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    Path(str(args.output_prefix) + ".md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
