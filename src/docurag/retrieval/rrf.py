from uuid import UUID


def reciprocal_rank_fusion(
    rankings: list[list[UUID]],
    *,
    constant: int = 60,
) -> dict[UUID, float]:
    if constant < 1:
        raise ValueError("RRF constant must be positive.")

    scores: dict[UUID, float] = {}

    for ranking in rankings:
        seen: set[UUID] = set()

        for rank, chunk_id in enumerate(ranking, start=1):
            if chunk_id in seen:
                continue

            seen.add(chunk_id)

            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (constant + rank)

    return scores
