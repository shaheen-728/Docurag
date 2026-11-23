import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path("results/day5/baseline")
ANSWER_PATH = ROOT / "answers.json"
JUDGMENT_PATH = ROOT / "judgments-dev80.json"


def average(values: list) -> float | None:
    known = [value for value in values if value is not None]
    return round(mean(known), 4) if known else None


def display(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()

    heldout = args.split == "test"
    root = Path("results/day5/heldout") if heldout else ROOT
    answer_path = root / "answers.json"
    judgment_path = root / "judgments.json" if heldout else JUDGMENT_PATH

    answer_raw = answer_path.read_bytes()
    answers = json.loads(answer_raw)
    judgments = json.loads(judgment_path.read_text(encoding="utf-8"))

    dataset_directory = Path("data/eval/v1")
    dataset_raw = (dataset_directory / f"{args.split}.json").read_bytes()
    dataset = json.loads(dataset_raw)
    dataset_hash = hashlib.sha256(dataset_raw).hexdigest()
    manifest = json.loads(
        (dataset_directory / "manifest.json").read_text(encoding="utf-8")
    )

    if manifest.get(f"{args.split}_sha256") != dataset_hash:
        raise ValueError("Dataset differs from its frozen manifest.")

    if (
        answers["identity"]["dataset_sha256"] != dataset_hash
        or judgments["identity"]["dataset_sha256"] != dataset_hash
    ):
        raise ValueError("Answers or judgments use another dataset.")

    dataset_ids = {item["id"] for item in dataset}
    expected_count = 20 if heldout else 80

    if (
        len(dataset) != expected_count
        or len(dataset_ids) != expected_count
        or set(answers["answers"]) != dataset_ids
    ):
        raise ValueError("Expected the complete frozen question set.")

    modes = answers["identity"]["modes"]

    if heldout and modes != {"vector": {"mode": "vector", "rerank": False}}:
        raise ValueError("Held-out evaluation must be vector-only.")

    for qid, saved in answers["answers"].items():
        if set(saved) != set(modes):
            raise ValueError(f"Incomplete answer modes: {qid}")

        if any(item["completed"] is not True for item in saved.values()):
            raise ValueError(f"Incomplete answer: {qid}")

    review_path = root / (
        "relevance_review.json" if heldout else "relevance_review.full.json"
    )

    if (
        hashlib.sha256(review_path.read_bytes()).hexdigest()
        != judgments["identity"]["review_sha256"]
    ):
        raise ValueError("Judgments use another relevance review.")

    if (
        hashlib.sha256(answer_raw).hexdigest()
        != (judgments["identity"]["answers_sha256"])
    ):
        raise ValueError("Judgments belong to a different answers file.")

    modes = list(answers["identity"]["modes"])
    expected_ids = set(answers["answers"])

    expected_pairs = {
        (qid, mode) for qid, saved in answers["answers"].items() for mode in saved
    }
    actual_pairs = {
        (qid, mode) for qid, saved in judgments["judgments"].items() for mode in saved
    }

    if not actual_pairs <= expected_pairs:
        raise ValueError("Unexpected judgments.")

    # Compare every mode on exactly the same completed questions.
    complete_ids = [
        qid
        for qid in answers["answers"]
        if set(judgments["judgments"].get(qid, {})) == set(modes)
    ]

    complete = actual_pairs == expected_pairs

    lines = [
        (
            "# Held-out answer evaluation"
            if heldout
            else "# Development answer evaluation"
        ),
        "",
        f"Status: {'complete' if complete else 'PARTIAL — provisional'}",
        f"Saved judgments: {len(actual_pairs)}/{len(expected_pairs)}",
        f"Comparable questions: {len(complete_ids)}/{len(expected_ids)}",
        "",
    ]

    rows = []

    if complete_ids:
        lines.extend(
            [
                "| Mode | Correctness | Relevancy | Faithfulness | "
                "Faithfulness N | Context precision | Precision N | "
                "Abstention rate | Empty-context rate |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )

        for mode in modes:
            items = [judgments["judgments"][qid][mode] for qid in complete_ids]

            row = {
                "mode": mode,
                "questions": len(items),
                "correctness": average([item["answer_correctness"] for item in items]),
                "relevancy": average([item["answer_relevancy"] for item in items]),
                "faithfulness": average([item["faithfulness"] for item in items]),
                "faithfulness_n": sum(
                    item["faithfulness"] is not None for item in items
                ),
                "context_precision": average(
                    [item["context_precision"] for item in items]
                ),
                "context_precision_n": sum(
                    item["context_precision"] is not None for item in items
                ),
                "abstention_rate": average([int(item["abstained"]) for item in items]),
                "empty_context_rate": average(
                    [int(item["returned_context_count"] == 0) for item in items]
                ),
            }
            rows.append(row)

            lines.append(
                f"| {mode} "
                f"| {display(row['correctness'])} "
                f"| {display(row['relevancy'])} "
                f"| {display(row['faithfulness'])} "
                f"| {row['faithfulness_n']} "
                f"| {display(row['context_precision'])} "
                f"| {row['context_precision_n']} "
                f"| {display(row['abstention_rate'])} "
                f"| {display(row['empty_context_rate'])} |"
            )
    else:
        lines.append("No questions have judgments for every mode yet.")

    lines.extend(
        [
            "",
            "Scores are macro averages across the comparable questions.",
            "Null faithfulness and context precision are excluded.",
            "Their denominators are shown in the table.",
            "Context precision = relevant chunks / returned chunks.",
            "Relevance labels are assistant-reviewed pooled labels.",
            "Scores use a custom same-model LLM judge, not RAGAS.",
            "Calibration: 6/6 sanity checks passed; accuracy remains unproven.",
            (
                "This is the frozen held-out test of the selected configuration."
                if heldout
                else "This is development evaluation, not the held-out test."
            ),
            "Retry/parser policy changes are documented in DECISIONS.md.",
        ]
    )

    report = {
        "split": args.split,
        "complete": complete,
        "saved_judgments": len(actual_pairs),
        "expected_judgments": len(expected_pairs),
        "comparable_question_ids": complete_ids,
        "judge_identity": judgments["identity"],
        "rows": rows,
    }

    # Partial reports never overwrite the completed report.
    name = "answer-metrics" if complete else "answer-metrics.partial"

    (root / f"{name}.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / f"{name}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
