import argparse
import asyncio
import hashlib
import inspect
import json
from contextlib import aclosing
from pathlib import Path
from time import perf_counter

from docurag.chat.prompts import (
    ANSWER_RULES,
    INSUFFICIENT_CONTEXT,
    build_answer_prompt,
)
from docurag.core.config import get_settings
from docurag.llm.factory import build_provider
from docurag.schemas.search import SearchHit


def checksum(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


async def run(args: argparse.Namespace) -> None:
    run_raw = args.run.read_bytes()
    retrieval = json.loads(run_raw)

    dataset_raw = args.dataset.read_bytes()
    questions = json.loads(dataset_raw)
    dataset_hash = checksum(dataset_raw)

    if args.dataset.stem not in {"dev", "test"}:
        raise ValueError("Use a frozen development or test dataset.")

    if args.dataset.stem == "test":
        if not args.allow_heldout:
            raise ValueError("Test generation requires --allow-heldout.")

        if len(questions) != 20 or args.limit != 20:
            raise ValueError("The held-out run requires all 20 questions.")

        expected_modes = {"vector": {"mode": "vector", "rerank": False}}

        if retrieval["identity"]["modes"] != expected_modes:
            raise ValueError("Held-out retrieval must be vector-only.")

        expected_directory = Path("results/day5/heldout").resolve()

        if args.output.resolve().parent != expected_directory:
            raise ValueError("Save held-out answers under results/day5/heldout.")

    manifest = json.loads(
        (args.dataset.parent / "manifest.json").read_text(
            encoding="utf-8",
        )
    )

    if manifest.get(f"{args.dataset.stem}_sha256") != dataset_hash:
        raise ValueError("Dataset differs from the frozen manifest.")

    if retrieval["identity"]["dataset_sha256"] != dataset_hash:
        raise ValueError("Retrieval run belongs to another dataset.")

    if len({item["id"] for item in questions}) != len(questions):
        raise ValueError("Duplicate question IDs.")

    settings = get_settings()
    model_field = f"{settings.llm_provider}_generation_model"
    model = getattr(settings, model_field)

    # Pin the prompt rules and builder so changed prompts cannot be
    # silently mixed into a resumed answer run.
    prompt_version = checksum(
        (ANSWER_RULES + "\n" + inspect.getsource(build_answer_prompt)).encode()
    )

    identity = {
        "retrieval_run_sha256": checksum(run_raw),
        "dataset_sha256": dataset_hash,
        "provider": settings.llm_provider,
        "model": model,
        "prompt_sha256": prompt_version,
        "generation_method": "provider.stream",
        "fallback_enabled": False,
        "timeout_seconds": args.timeout,
        "modes": retrieval["identity"]["modes"],
    }

    if args.dataset.stem == "test":
        baseline_path = Path("results/day5/baseline/answers.json")
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

        for key in (
            "provider",
            "model",
            "prompt_sha256",
            "generation_method",
            "fallback_enabled",
        ):
            if baseline["identity"][key] != identity[key]:
                raise ValueError(f"Held-out generation differs from baseline: {key}")

    if args.output.exists():
        state = json.loads(args.output.read_text(encoding="utf-8"))

        if state["identity"] != identity:
            raise ValueError("Evaluation configuration changed. Use a new output file.")
    else:
        state = {
            "identity": identity,
            "dataset_path": str(args.dataset),
            "retrieval_run_path": str(args.run),
            "answers": {},
            "errors": [],
        }
        save_json(args.output, state)

    selected = questions[: args.limit]
    modes = identity["modes"]

    # Validate all required retrieval results before making API calls.
    for question in selected:
        saved = retrieval["responses"].get(question["id"], {})

        if set(saved) != set(modes):
            raise ValueError(f"Incomplete retrieval results for {question['id']}.")

    print(
        f"Questions: {len(selected)}; "
        f"maximum new answers: {len(selected) * len(modes)}",
        flush=True,
    )
    print(f"Fixed model: {settings.llm_provider}/{model}", flush=True)

    for question in selected:
        question_id = question["id"]
        answers = state["answers"].setdefault(question_id, {})

        for mode in modes:
            if mode in answers:
                print(f"{question_id} {mode}: reused", flush=True)
                continue

            result = retrieval["responses"][question_id][mode]["response"]
            hits = [SearchHit.model_validate(hit) for hit in result["results"]]

            prompt, citations = build_answer_prompt(
                question["question"],
                hits,
            )

            print(f"{question_id} {mode}: generating", flush=True)
            started = perf_counter()
            first_token_ms = None
            fragments = []
            provider = None

            try:
                if hits:
                    provider = build_provider(settings)

                    async with asyncio.timeout(args.timeout):
                        async with aclosing(
                            provider.stream(
                                prompt,
                                system_instruction=ANSWER_RULES,
                            )
                        ) as stream:
                            async for fragment in stream:
                                if not fragment:
                                    continue

                                if first_token_ms is None:
                                    first_token_ms = round(
                                        (perf_counter() - started) * 1000,
                                        3,
                                    )

                                fragments.append(fragment)

                    answer = "".join(fragments).strip()

                    if not answer:
                        raise RuntimeError("Model returned no answer.")
                else:
                    # No retrieved evidence: deterministic abstention.
                    # Do not spend an API call on an empty context.
                    answer = INSUFFICIENT_CONTEXT

                answers[mode] = {
                    "question": question["question"],
                    "reference_answer": question["reference_answer"],
                    "answer": answer,
                    "context": [hit.model_dump(mode="json") for hit in hits[:5]],
                    # These are available sources, not verified citations.
                    "available_citations": [
                        citation.model_dump(mode="json") for citation in citations
                    ],
                    "prompt_sha256": checksum(prompt.encode()),
                    "generation_called": bool(hits),
                    "generation_model": model if hits else None,
                    "first_token_ms": first_token_ms,
                    "generation_latency_ms": round(
                        (perf_counter() - started) * 1000,
                        3,
                    ),
                    "completed": True,
                }

                save_json(args.output, state)
                print(
                    f"  saved; characters={len(answer)}",
                    flush=True,
                )

            except Exception as error:
                state["errors"].append(
                    {
                        "question_id": question_id,
                        "mode": mode,
                        "error_type": type(error).__name__,
                        "partial_output_characters": sum(
                            len(fragment) for fragment in fragments
                        ),
                    }
                )
                save_json(args.output, state)
                raise

            finally:
                if provider is not None:
                    await provider.aclose()

            await asyncio.sleep(args.pause)

    print(f"Saved: {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        type=Path,
        default=Path("results/day5/baseline/run.json"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/eval/v1/dev.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/day5/baseline/answers.json"),
    )
    parser.add_argument("--allow-heldout", action="store_true")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--pause", type=float, default=8)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()

    if args.limit < 1 or args.pause < 0 or args.timeout <= 0:
        parser.error("Limit and timeout must be positive; pause cannot be negative.")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
