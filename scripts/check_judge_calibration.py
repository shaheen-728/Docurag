import argparse
import asyncio
import hashlib
import inspect
import json
from pathlib import Path

from docurag.core.config import get_settings
from docurag.llm.factory import build_provider
from scripts.judge_eval_answers import (
    RULES,
    Judgment,
    judge_with_retry,
    parse_judgment,
    save_json,
)


def checksum(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


async def run(args: argparse.Namespace) -> None:
    raw = args.input.read_bytes()
    cases = json.loads(raw)
    dataset_raw = args.dataset.read_bytes()
    questions = {item["id"]: item for item in json.loads(dataset_raw)}

    if len(cases) != 6 or len({item["id"] for item in cases}) != 6:
        raise ValueError("Expected six unique calibration cases.")

    for qid in ("q001", "q002", "q005"):
        pair = [item for item in cases if item["question_id"] == qid]

        if len(pair) != 2 or {item["variant"] for item in pair} != {
            "original",
            "deliberately_wrong",
        }:
            raise ValueError(f"Invalid pair: {qid}")

        if pair[0]["context"] != pair[1]["context"]:
            raise ValueError(f"Contexts differ: {qid}")

    payloads = {}

    for item in cases:
        question = questions[item["question_id"]]

        if item["question"] != question["question"]:
            raise ValueError("Question text differs from dev dataset.")

        payloads[item["id"]] = {
            "question": item["question"],
            "reference_answer": question["reference_answer"],
            "reference_evidence": question["source_content"],
            "answer": item["answer"],
            "retrieved_context": [
                {
                    "source_number": number,
                    "content": hit["content"],
                }
                for number, hit in enumerate(item["context"], start=1)
            ],
        }

    print(f"Validated {len(payloads)} calibration payloads.")

    if not args.execute:
        print("DRY RUN: no API calls made.")
        return

    settings = get_settings()
    model = getattr(
        settings,
        f"{settings.llm_provider}_generation_model",
    )

    system_instruction = (
        RULES + "\nOutput JSON schema:\n" + json.dumps(Judgment.model_json_schema())
    )

    identity = {
        "cases_sha256": checksum(raw),
        "dataset_sha256": checksum(dataset_raw),
        "provider": settings.llm_provider,
        "model": model,
        "rubric_sha256": checksum(system_instruction.encode()),
        "implementation_sha256": checksum(
            (
                inspect.getsource(parse_judgment) + inspect.getsource(judge_with_retry)
            ).encode()
        ),
        "timeout_seconds": args.timeout,
    }

    if args.output.exists():
        state = json.loads(args.output.read_text())

        if state["identity"] != identity:
            raise ValueError(
                "Calibration configuration changed. Use a new output path."
            )
    else:
        state = {"identity": identity, "results": {}}
        save_json(args.output, state)

    provider = build_provider(settings)

    try:
        for item in cases:
            case_id = item["id"]

            if case_id in state["results"]:
                print(f"{case_id}: reused", flush=True)
                continue

            print(f"{case_id}: judging", flush=True)

            judgment = await judge_with_retry(
                provider,
                payloads[case_id],
                system_instruction=system_instruction,
                context_count=len(item["context"]),
                timeout=args.timeout,
                pause=args.pause,
            )

            claims = judgment.claims
            faithfulness = (
                sum(claim.supported for claim in claims) / len(claims)
                if claims
                else None
            )

            if item["variant"] == "deliberately_wrong":
                passed = (
                    judgment.answer_correctness == 0.0
                    and not judgment.abstained
                    and any(not claim.supported for claim in claims)
                )
            else:
                passed = (
                    judgment.answer_correctness == 1.0
                    and not judgment.abstained
                    and faithfulness == 1.0
                )

            state["results"][case_id] = {
                **judgment.model_dump(),
                "faithfulness": faithfulness,
                "passed": passed,
            }
            save_json(args.output, state)

            print(f"  passed={passed}", flush=True)
            await asyncio.sleep(args.pause)

    finally:
        await provider.aclose()

    passed_count = sum(item["passed"] for item in state["results"].values())
    print(f"Calibration checks passed: {passed_count}/{len(cases)}")
    print(f"Saved: {args.output}")
    print("Six cases are a basic sanity check, not proof of judge accuracy.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/eval/judge_calibration.json"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/eval/v1/dev.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/day5/judge-calibration.json"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--pause", type=float, default=8)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()

    if args.pause < 0 or args.timeout <= 0:
        parser.error("Pause cannot be negative; timeout must be positive.")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
