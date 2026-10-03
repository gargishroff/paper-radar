"""Pure-function tests for clustering, candidates, slate building, explanations and tokens."""

import numpy as np
import pytest

from radar.explain.explain import EXPLORE_WHY, ExplainInput, explain
from radar.feedback import sign, verify
from radar.profile.cluster import choose_clusters, medoid, normalize_weights, recency_weights
from radar.rank.candidates import generate
from radar.rank.slate import Scored, allocate, build_slate
from radar.usage import BudgetExceeded


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _blobs(rng, centres, n_each, noise=0.05):
    return _unit(np.vstack([c + noise * rng.standard_normal((n_each, len(c))) for c in centres]))


def test_recency_weights():
    w = recency_weights(np.array([0.0, 365.0, -5.0]), 365.0)
    assert w[0] == 1.0 and w[2] == 1.0  # future dates clip to age 0
    assert w[1] == pytest.approx(np.exp(-1))


def test_clustering_separates_two_topics():
    rng = np.random.default_rng(0)
    d = 16
    a, b = np.eye(d)[0], np.eye(d)[1]
    x = _blobs(rng, [a, b], 5)
    result = choose_clusters(x, np.ones(len(x)), k_max=4)
    assert len(result.medoids) == 2
    assert len(set(result.labels[:5])) == 1 and len(set(result.labels[5:])) == 1
    assert result.labels[0] != result.labels[5]


def test_small_profile_groups_only_similar_papers():
    # Papers 0 and 1 are near-duplicates; paper 2 is unrelated (cosine 0).
    x = _unit(np.array([[1.0, 0.1, 0], [1.0, -0.1, 0], [0, 0, 1.0]]))
    result = choose_clusters(x, np.ones(3), k_max=6)
    assert result.labels.tolist() == [0, 0, 1] and len(result.medoids) == 2
    # Capped at k_max interests.
    assert choose_clusters(x, np.ones(3), k_max=1).labels.tolist() == [0, 0, 0]


def test_medoid_is_the_central_real_paper():
    x = _unit(np.array([[1, 0.0], [1, 0.1], [1, -0.1], [0, 1.0]]))
    assert medoid(x, np.ones(4), np.array([True, True, True, False])) == 0


def test_normalize_weights():
    assert normalize_weights(np.array([1.0, 3.0])).tolist() == [0.25, 0.75]
    assert normalize_weights(np.array([0.0, 0.0])).tolist() == [0.5, 0.5]


def test_candidates_topm_and_explore_band():
    rng = np.random.default_rng(0)
    papers = _unit(rng.standard_normal((400, 8)))
    interests = _unit(rng.standard_normal((2, 8)))
    c = generate(papers, interests, top_m=50, band=(50, 300), pool_size=20, rng=rng)
    sims = papers @ interests.T
    assert list(c.per_interest[0]) == list(np.argsort(-sims[:, 0])[:50])
    assert len(c.explore) == 20
    top = set(c.per_interest[0]) | set(c.per_interest[1])
    assert top.isdisjoint(c.explore)  # near misses only, never a top-M paper


def test_allocate_proportional_with_minimum_one():
    slots = allocate(np.array([0.7, 0.2, 0.1]), np.array([50, 50, 50]), 10)
    assert slots.sum() == 10 and slots.min() >= 1 and slots[0] > slots[1] >= slots[2]


def test_allocate_respects_available_and_more_interests_than_slots():
    assert allocate(np.array([0.8, 0.2]), np.array([2, 50]), 10).tolist() == [2, 8]
    assert allocate(np.array([0.1, 0.5, 0.4]), np.array([5, 5, 5]), 2).tolist() == [0, 1, 1]
    assert allocate(np.array([0.5, 0.5]), np.array([0, 3]), 5).tolist() == [0, 3]


def test_build_slate_shape_order_and_no_duplicates():
    rng = np.random.default_rng(0)
    vectors = _unit(rng.standard_normal((100, 8)))
    scored = [Scored(i, 0, 1 - i / 100) for i in range(40)]
    scored += [Scored(i, 1, 1 - i / 100) for i in range(40, 80)]
    rule = [Scored(95, 0, 0.1)]
    slate = build_slate(
        scored,
        np.array([0.5, 0.5]),
        vectors,
        feed_size=12,
        explore=[90, 91, 92],
        explore_slots=2,
        rule=rule,
        lam=0.7,
    )
    indices = [s.index for s in slate]
    assert len(slate) == 12 and len(set(indices)) == 12
    assert slate[0].kind == "rule"
    assert [s.kind for s in slate].count("explore") == 2
    main = [s for s in slate if s.kind == "main"]
    assert [s.interest for s in main[:4]] == [0, 1, 0, 1]  # interests interleave


class _Chat:
    model = "fake"

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, 0

    def chat_json(self, system, user, schema, essential=False):
        self.calls += 1
        if self.error:
            raise self.error
        return self.reply


_ITEMS = [
    ExplainInput("A", "Paper A", "Does X. Then Y.", [("P1", "My paper")], ("Topic", "Papers on X")),
    ExplainInput("B", "Paper B", "Does Z.", [], None, explore=True),
]


def test_explanation_guardrail():
    chat = _Chat(
        {
            "papers": [
                {"arxiv_id": "A", "tldr": "T", "why": "Invented link", "relates_to": "NOT-MINE"},
                {"arxiv_id": "B", "tldr": "TB", "why": "forced", "relates_to": "interest"},
            ]
        }
    )
    out = explain(chat, _ITEMS)
    assert out["A"].tldr == "T" and not out["A"].grounded
    assert out["A"].why == "Similar to: My paper"
    assert out["B"].tldr == "TB" and out["B"].why == EXPLORE_WHY


def test_explanation_accepts_listed_paper_or_interest():
    for ref in ("P1", "interest"):
        chat = _Chat({"papers": [{"arxiv_id": "A", "tldr": "T", "why": "W", "relates_to": ref}]})
        assert explain(chat, _ITEMS[:1])["A"].grounded


def test_explanation_falls_back_when_over_budget():
    out = explain(_Chat(error=BudgetExceeded("over")), _ITEMS)
    assert out["A"].tldr == "Does X." and out["B"].why == EXPLORE_WHY


def test_signed_tokens_roundtrip_and_reject_tampering():
    token = sign("secret", 3, "2401.00001", "up", 7)
    assert verify("secret", token) == (3, "2401.00001", "up", 7)
    assert verify("other-secret", token) is None
    assert verify("secret", token[:-2] + "xx") is None
    assert verify("secret", "garbage") is None
    assert verify("secret", sign("secret", 1, "cs/0112017", "save", None)) == (
        1,
        "cs/0112017",
        "save",
        None,
    )
