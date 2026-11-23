import hashlib
import json
from pathlib import Path
from statistics import mean

DIRECTORY = Path("results/day5/baseline")
RUN_PATH = DIRECTORY / "run.json"
REVIEW_PATH = DIRECTORY / "relevance_review.full.json"


def main() -> None:
    run_raw = RUN_PATH.read_bytes()
    review_raw = REVIEW_PATH.read_bytes()
    run = json.loads(run_raw)
    review = json.loads(review_raw)

    if hashlib.sha256(run_raw).hexdigest() != review["run_sha256"]:
        raise ValueError("Review belongs to a different run.")

    if run["identity"]["dataset_sha256"] != review["dataset_sha256"]:
        raise ValueError("Dataset checksums do not match.")

    questions = review["questions"]
    question_ids = [question["id"] for question in questions]

    if not questions or len(set(question_ids)) != len(question_ids):
        raise ValueError("Review must contain unique question IDs.")

    if set(question_ids) != set(run["responses"]):
        raise ValueError("Review must cover every question in the run.")

    modes = run["identity"]["modes"]
    k = run["identity"]["top_k"]

    for question in questions:
        if not question["reviewed"]:
            raise ValueError(f"{question['id']} needs review.")

        relevant = set(question["relevant_chunk_ids"])
        candidates = {item["chunk_id"] for item in question["candidates"]}

        if not relevant or not relevant <= candidates:
            raise ValueError(f"Invalid relevance labels for {question['id']}.")

        responses = run["responses"][question["id"]]

        if set(responses) != set(modes):
            raise ValueError(f"Incomplete methods for {question['id']}.")

        for saved in responses.values():
            returned = {hit["chunk_id"] for hit in saved["response"]["results"]}
            if not returned <= candidates:
                raise ValueError("Returned chunks are missing from review.")

    rows = []
    details = []

    for mode in modes:
        hits = []
        reciprocal_ranks = []
        recalls = []
        latencies = []

        for question in questions:
            saved = run["responses"][question["id"]][mode]
            result = saved["response"]
            retrieved = [hit["chunk_id"] for hit in result["results"][:k]]
            relevant = set(question["relevant_chunk_ids"])

            first_rank = next(
                (
                    rank
                    for rank, chunk_id in enumerate(retrieved, start=1)
                    if chunk_id in relevant
                ),
                None,
            )

            hit = int(first_rank is not None)
            reciprocal_rank = 1 / first_rank if first_rank else 0.0
            recall = len(set(retrieved) & relevant) / len(relevant)

            hits.append(hit)
            reciprocal_ranks.append(reciprocal_rank)
            recalls.append(recall)
            latencies.append(saved["client_latency_ms"])

            details.append(
                {
                    "question_id": question["id"],
                    "mode": mode,
                    "hit": hit,
                    "reciprocal_rank": reciprocal_rank,
                    "pooled_label_recall": recall,
                }
            )

        rows.append(
            {
                "mode": mode,
                "questions": len(questions),
                "hit_at_k": mean(hits),
                "mrr_at_k": mean(reciprocal_ranks),
                "pooled_label_recall_at_k": mean(recalls),
                "mean_client_latency_ms": mean(latencies),
            }
        )

    output = {
        "run_sha256": hashlib.sha256(run_raw).hexdigest(),
        "review_sha256": hashlib.sha256(review_raw).hexdigest(),
        "top_k": k,
        "label_provenance": "assistant-reviewed",
        "label_scope": "pooled_top5_plus_generation_source",
        "rows": rows,
        "per_question": details,
    }

    (DIRECTORY / "metrics.json").write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Development retrieval pilot",
        "",
        f"Questions: {len(questions)}. Top k: {k}.",
        "",
        f"| Mode | Hit@{k} | MRR@{k} | Pooled Recall@{k} | Mean client ms |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            f"| {row['mode']} "
            f"| {row['hit_at_k']:.3f} "
            f"| {row['mrr_at_k']:.3f} "
            f"| {row['pooled_label_recall_at_k']:.3f} "
            f"| {row['mean_client_latency_ms']:.1f} |"
        )

    lines.extend(
        [
            "",
            "Labels are assistant-reviewed, not human-verified.",
            "Recall is measured against pooled labels, not all relevant corpus chunks.",
            "Latency includes cold-start effects; no p95 claim is made.",
            "This development pilot is not the held-out test.",
        ]
    )

    markdown = "\n".join(lines) + "\n"
    (DIRECTORY / "metrics.md").write_text(
        markdown,
        encoding="utf-8",
    )
    print(markdown)


if __name__ == "__main__":
    main()
