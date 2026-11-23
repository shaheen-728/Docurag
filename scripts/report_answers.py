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
    answer_raw = ANSWER_PATH.read_bytes()
    answers = json.loads(answer_raw)
    judgments = json.loads(JUDGMENT_PATH.read_text(encoding="utf-8"))

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
        "# Development answer evaluation",
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
            "This is development evaluation, not the held-out test.",
            "Retry/parser policy changes are documented in DECISIONS.md.",
        ]
    )

    report = {
        "complete": complete,
        "saved_judgments": len(actual_pairs),
        "expected_judgments": len(expected_pairs),
        "comparable_question_ids": complete_ids,
        "judge_identity": judgments["identity"],
        "rows": rows,
    }

    # Partial reports never overwrite the completed report.
    name = "answer-metrics" if complete else "answer-metrics.partial"

    (ROOT / f"{name}.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    (ROOT / f"{name}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
