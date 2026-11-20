import argparse
import json
import time
from pathlib import Path

import httpx

QUESTIONS_PATH = Path("scripts/questions.json")
OUTPUT_DIRECTORY = Path("results/day3")
RUN_PATH = OUTPUT_DIRECTORY / "run.json"
REVIEW_PATH = OUTPUT_DIRECTORY / "review.json"

MODES = {
    "keyword": {"mode": "keyword", "rerank": False},
    "vector": {"mode": "vector", "rerank": False},
    "hybrid": {"mode": "hybrid", "rerank": False},
    "hybrid_rerank": {"mode": "hybrid", "rerank": True},
}


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def collect() -> None:
    questions = load_questions()

    # Resume an interrupted run without repeating successful requests.
    if RUN_PATH.exists():
        run = json.loads(RUN_PATH.read_text(encoding="utf-8"))

        if run["questions"] != questions:
            raise ValueError(
                "Questions changed. Archive results/day3 before "
                "starting a new comparison."
            )
    else:
        run = {
            "questions": questions,
            "top_k": 5,
            "candidate_k": 20,
            "responses": {},
        }

    with httpx.Client(timeout=300) as client:
        for question in questions:
            question_id = question["id"]

            responses = run["responses"].setdefault(
                question_id,
                {},
            )

            for name, options in MODES.items():
                if name in responses:
                    print(f"{question_id} {name}: reused")
                    continue

                print(
                    f"{question_id} {name}: requesting",
                    flush=True,
                )

                response = client.post(
                    "http://127.0.0.1:8000/search",
                    json={
                        "query": question["query"],
                        "top_k": 5,
                        "candidate_k": 20,
                        **options,
                    },
                )

                if response.is_error:
                    save_json(RUN_PATH, run)

                    raise RuntimeError(
                        f"{question_id} {name}: "
                        f"HTTP {response.status_code}: "
                        f"{response.text}"
                    )

                result = response.json()

                if result["mode"] != options["mode"]:
                    raise RuntimeError("Unexpected retrieval mode.")

                if (
                    name == "hybrid_rerank"
                    and result["results"]
                    and not result["rerank_applied"]
                ):
                    raise RuntimeError("Reranking was not applied.")

                responses[name] = result
                save_json(RUN_PATH, run)

                print(
                    f"  results={len(result['results'])} "
                    f"latency_ms={result['latency_ms']}",
                    flush=True,
                )

                # Modest spacing between query-embedding requests.
                time.sleep(2)

    if REVIEW_PATH.exists():
        print(f"Existing {REVIEW_PATH} preserved. Review it against the saved run.")
        return

    review = {}

    for question in questions:
        pooled = {}

        for response in run["responses"][question["id"]].values():
            for hit in response["results"]:
                pooled.setdefault(
                    hit["chunk_id"],
                    {
                        "chunk_id": hit["chunk_id"],
                        "filename": hit["filename"],
                        "section_heading": hit["metadata"].get("section_heading"),
                        "content": hit["content"],
                    },
                )

        review[question["id"]] = {
            "query": question["query"],
            "reviewed": False,
            "relevant_chunk_ids": [],
            "notes": "",
            # Order by ID rather than a particular mode's ranking.
            "candidates": sorted(
                pooled.values(),
                key=lambda hit: hit["chunk_id"],
            ),
        }

    save_json(REVIEW_PATH, review)

    print(f"\nResponses saved to {RUN_PATH}")
    print(f"Human-review file saved to {REVIEW_PATH}")


def report() -> None:
    questions = load_questions()
    run = json.loads(RUN_PATH.read_text(encoding="utf-8"))
    review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))

    if run["questions"] != questions:
        raise ValueError("Questions differ from the collected run.")

    for question in questions:
        question_id = question["id"]

        if not review[question_id]["reviewed"]:
            raise ValueError(f"{question_id} still needs human review.")

        if review[question_id]["query"] != question["query"]:
            raise ValueError(f"{question_id} review query differs.")

        if set(run["responses"][question_id]) != set(MODES):
            raise ValueError(f"{question_id} has incomplete responses.")

    lines = [
        "# Day 3 Retrieval Comparison",
        "",
        f"Questions: {len(questions)}. Top k: 5. Candidates per method: 20.",
        "",
        "| Mode | Hit@5 | MRR@5 | Mean request latency ms |",
        "| --- | ---: | ---: | ---: |",
    ]

    for mode in MODES:
        hits = []
        reciprocal_ranks = []
        latencies = []

        for question in questions:
            question_id = question["id"]

            relevant = set(review[question_id]["relevant_chunk_ids"])

            response = run["responses"][question_id][mode]
            first_relevant_rank = None

            for rank, hit in enumerate(
                response["results"][:5],
                start=1,
            ):
                if hit["chunk_id"] in relevant:
                    first_relevant_rank = rank
                    break

            hits.append(int(first_relevant_rank is not None))

            reciprocal_ranks.append(
                1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
            )

            latencies.append(response["latency_ms"])

        count = len(questions)

        lines.append(
            f"| {mode} "
            f"| {sum(hits) / count:.3f} "
            f"| {sum(reciprocal_ranks) / count:.3f} "
            f"| {sum(latencies) / count:.1f} |"
        )

    lines.extend(
        [
            "",
            "Assistant-reviewed relevance labels use the pooled returned chunks.",
            "This is a ten-question development check, not a held-out benchmark.",
            "Latency may include model loading and other cold-start effects.",
        ]
    )

    output = "\n".join(lines) + "\n"

    (OUTPUT_DIRECTORY / "summary.md").write_text(
        output,
        encoding="utf-8",
    )

    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "action",
        choices=["collect", "report"],
    )

    args = parser.parse_args()

    if args.action == "collect":
        collect()
    else:
        report()


if __name__ == "__main__":
    main()
