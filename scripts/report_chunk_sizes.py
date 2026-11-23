"""Score a frozen chunk-size retrieval run against its reviewed pool; no API calls."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

VARIANTS = ("baseline", "token256", "token512", "token1024")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: Path) -> tuple[dict | list, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def score(run: dict, review: dict, dataset: list, dataset_hash: str) -> dict:
    identity = run["identity"]
    require(identity["dataset_sha256"] == dataset_hash, "Run dataset hash mismatch.")
    require(review["dataset_sha256"] == dataset_hash, "Review dataset hash mismatch.")
    require(
        identity["mode"] == "vector" and identity["rerank"] is False,
        "Expected vector retrieval without reranking.",
    )
    require(identity["top_k"] == 5, "Expected top_k=5.")
    require(set(identity["variants"]) == set(VARIANTS), "Unexpected variants.")
    require(
        review["label_scope"] == "pooled_top5_plus_source_and_exact_evidence",
        "Unexpected relevance pool.",
    )
    questions = {item["id"]: item for item in dataset}
    require(
        len(questions) == len(dataset) == 80,
        "Expected 80 unique development questions.",
    )
    require(set(run["responses"]) == set(questions), "Incomplete run question set.")
    reviews = {item["id"]: item for item in review["questions"]}
    require(
        len(reviews) == len(review["questions"]) and set(reviews) == set(questions),
        "Incomplete or duplicate reviewed questions.",
    )
    rows = []
    consistency = {}
    for qid, question in questions.items():
        assessed = reviews[qid]
        require(
            assessed["reviewed"] is True and bool(assessed["reviewer"]),
            f"{qid}: question is not reviewed.",
        )
        require(
            assessed["question"] == question["question"]
            and assessed["reference_answer"] == question["reference_answer"]
            and assessed["evidence_text"] == question["evidence_text"],
            f"{qid}: question/reference/evidence changed.",
        )
        candidates = {item["chunk_id"]: item for item in assessed["candidates"]}
        require(
            len(candidates) == len(assessed["candidates"]),
            f"{qid}: duplicate candidates.",
        )
        require(
            set(run["responses"][qid]) == set(VARIANTS), f"{qid}: incomplete variants."
        )
        for candidate in candidates.values():
            require(
                type(candidate["relevant"]) is bool, f"{qid}: missing relevance label."
            )
            require(bool(candidate["review_notes"]), f"{qid}: missing review notes.")
            require(
                bool(candidate["variants"])
                and set(candidate["variants"]) <= set(VARIANTS),
                f"{qid}: invalid candidate variants.",
            )
            key = (qid, candidate["document_id"], candidate["content"])
            if key in consistency:
                require(
                    consistency[key] == candidate["relevant"],
                    f"{qid}: identical content has inconsistent relevance.",
                )
            consistency[key] = candidate["relevant"]
        for variant in VARIANTS:
            response = run["responses"][qid][variant]
            hits = response["results"]
            retrieved = [hit["chunk_id"] for hit in hits]
            require(
                len(retrieved) == len(set(retrieved)) == 5,
                f"{qid}/{variant}: expected five unique results.",
            )
            pool = {
                cid: item
                for cid, item in candidates.items()
                if variant in item["variants"]
            }
            require(
                set(retrieved) <= set(pool), f"{qid}/{variant}: unreviewed results."
            )
            for hit in hits:
                item = pool[hit["chunk_id"]]
                require(
                    all(
                        item[key] == hit[key]
                        for key in ("document_id", "filename", "content")
                    ),
                    f"{qid}/{variant}: reviewed content differs from result.",
                )
            relevant = {cid for cid, item in pool.items() if item["relevant"]}
            require(
                bool(relevant), f"{qid}/{variant}: no known positive in review pool."
            )
            found = set(retrieved) & relevant
            first_rank = next(
                (rank for rank, cid in enumerate(retrieved, 1) if cid in relevant), None
            )
            latency = response["retrieval_latency_ms"]
            require(
                type(latency) in (int, float)
                and math.isfinite(latency)
                and latency >= 0,
                f"{qid}/{variant}: invalid latency.",
            )
            rows.append(
                {
                    "question_id": qid,
                    "variant": variant,
                    "hit_at_5": float(bool(found)),
                    "reciprocal_rank_at_5": 1.0 / first_rank if first_rank else 0.0,
                    "pooled_recall_at_5": len(found) / len(relevant),
                    "precision_at_5": len(found) / 5,
                    "retrieved_relevant": len(found),
                    "pooled_relevant": len(relevant),
                    "first_relevant_rank": first_rank,
                    "retrieval_latency_ms": latency,
                }
            )
    metrics = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        metrics[variant] = {
            "questions": len(selected),
            **{
                name: mean(row[name] for row in selected)
                for name in (
                    "hit_at_5",
                    "reciprocal_rank_at_5",
                    "pooled_recall_at_5",
                    "precision_at_5",
                    "retrieval_latency_ms",
                )
            },
        }
    return {
        "complete": True,
        "question_count": 80,
        "metrics": metrics,
        "per_question": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=Path("results/day5/chunk-sizes/run-dev80.json")
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=Path("results/day5/chunk-sizes/relevance_review.dev80.json"),
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/eval/v1/dev.json"))
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/day5/chunk-sizes/metrics.dev80"),
    )
    args = parser.parse_args()
    run, run_hash = load(args.run)
    review, review_hash = load(args.review)
    dataset, dataset_hash = load(args.dataset)
    require(review["run_sha256"] == run_hash, "Review belongs to a different run file.")
    report = score(run, review, dataset, dataset_hash)
    report.update(
        {
            "run_sha256": run_hash,
            "review_sha256": review_hash,
            "dataset_sha256": dataset_hash,
            "label_scope": review["label_scope"],
            "label_provenance": review.get("label_provenance", "assistant-reviewed"),
            "human_verified": False,
            "latency_scope": run["latency_scope"],
            "retrieval_identity": run["identity"],
        }
    )
    lines = [
        "# Development chunk-size comparison",
        "",
        "Questions: 80. Vector retrieval. Top k: 5. No reranking.",
        "",
        "| Variant | Hit@5 | MRR@5 | Pooled Recall@5 | Precision@5 | Mean "
        "retrieval ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, values in report["metrics"].items():
        lines.append(
            f"| {variant} | {values['hit_at_5']:.3f} | "
            f"{values['reciprocal_rank_at_5']:.3f} | "
            f"{values['pooled_recall_at_5']:.3f} | "
            f"{values['precision_at_5']:.3f} | "
            f"{values['retrieval_latency_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Scores are macro averages over all 80 development questions.",
            "Labels are assistant-reviewed, not human-verified; partial "
            "answer support counts.",
            "Pooled recall uses variant-specific top-5, generation-source, "
            "and exact-evidence pools.",
            "It is not exhaustive corpus recall, and pools differ between "
            "chunking variants.",
            "Precision counts relevant retrieved chunks; it does not measure "
            "answer correctness.",
            "Latency includes database vector retrieval only; it excludes "
            "query embedding, HTTP,",
            "generation, and reranking. One fixed-order pass cannot establish "
            "a latency winner.",
            "This comparison does not evaluate generated answers"
            " or the held-out test set.",
            "No model API calls are made by this reporting script.",
            "",
            f"Run SHA256: {run_hash}",
            f"Review SHA256: {review_hash}",
        ]
    )
    markdown = "\n".join(lines) + "\n"
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.output_prefix) + ".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    Path(str(args.output_prefix) + ".md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
