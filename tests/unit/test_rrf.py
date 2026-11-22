from uuid import UUID

import pytest

from docurag.retrieval.rrf import reciprocal_rank_fusion

A = UUID(int=1)
B = UUID(int=2)
C = UUID(int=3)


def test_combines_support_from_both_rankings():
    scores = reciprocal_rank_fusion([[A, B], [B, C]])

    assert scores[A] == pytest.approx(1 / 61)
    assert scores[B] == pytest.approx(1 / 62 + 1 / 61)
    assert scores[C] == pytest.approx(1 / 62)
    assert max(scores, key=scores.get) == B


def test_duplicate_does_not_add_another_vote():
    scores = reciprocal_rank_fusion([[A, A], [A]])

    assert scores[A] == pytest.approx(2 / 61)


def test_empty_rankings_return_no_scores():
    assert reciprocal_rank_fusion([[], []]) == {}


def test_ranking_order_does_not_change_fusion_scores():
    first = reciprocal_rank_fusion([[A, B], [C, A]])
    second = reciprocal_rank_fusion([[C, A], [A, B]])

    assert first == pytest.approx(second)


@pytest.mark.parametrize("constant", [0, -1])
def test_rejects_nonpositive_constant(constant):
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion([[A]], constant=constant)
