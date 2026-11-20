import asyncio
from functools import lru_cache
from math import isfinite
from threading import Lock

from docurag.core.config import get_settings
from docurag.schemas.search import SearchHit

_model_lock = Lock()


class RerankerUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_model():
    # Import lazily so ordinary keyword search doesn't load PyTorch.
    import torch
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        get_settings().reranker_model,
        device="cpu",
        max_length=512,
        activation_fn=torch.nn.Identity(),
        trust_remote_code=False,
    )


def score_pairs(
    query: str,
    passages: list[str],
) -> list[float]:
    if not passages:
        return []

    # Serialize model loading and inference within this API process.
    with _model_lock:
        model = load_model()

        predictions = model.predict(
            [(query, passage) for passage in passages],
            batch_size=4,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    scores = [float(value) for value in predictions]

    if len(scores) != len(passages):
        raise RuntimeError("Unexpected number of reranker scores.")

    if not all(isfinite(score) for score in scores):
        raise RuntimeError("Reranker returned non-finite scores.")

    return scores


async def rerank_hits(
    query: str,
    hits: list[SearchHit],
) -> list[SearchHit]:
    if not hits:
        return []

    passages = []

    for hit in hits:
        heading = hit.metadata.get("section_heading") or ""

        passages.append(f"{heading}\n{hit.content}".strip())

    try:
        # Model loading and CPU inference must not run directly
        # on FastAPI's async event loop.
        scores = await asyncio.to_thread(
            score_pairs,
            query,
            passages,
        )
    except Exception as error:
        raise RerankerUnavailable("Reranker loading or inference failed.") from error

    scored_hits = [
        hit.model_copy(update={"rerank_score": score})
        for hit, score in zip(hits, scores, strict=True)
    ]

    ranked = sorted(
        scored_hits,
        key=lambda hit: (
            -hit.rerank_score,
            -hit.rrf_score,
            str(hit.chunk_id),
        ),
    )

    for rank, hit in enumerate(ranked, start=1):
        hit.rerank_rank = rank

    return ranked
