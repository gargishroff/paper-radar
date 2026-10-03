"""Stage 4: turn scored candidates into the day's list of papers."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Scored:
    index: int  # row in the day's paper matrix
    interest: int  # position of the matched interest
    score: float


@dataclass(frozen=True)
class Slot:
    index: int
    interest: int | None
    kind: str  # 'main' | 'explore' | 'rule'


def allocate(weights: np.ndarray, available: np.ndarray, n_slots: int) -> np.ndarray:
    """Slots per interest: one each, then extras to the interest furthest below its share."""
    slots = np.zeros(len(weights), dtype=int)
    active = np.flatnonzero(available > 0)
    if n_slots <= 0 or len(active) == 0:
        return slots
    active = active[np.argsort(-weights[active], kind="stable")][:n_slots]
    slots[active] = 1
    share = np.zeros(len(weights))
    share[active] = weights[active] / weights[active].sum() * n_slots
    for _ in range(n_slots - len(active)):
        room = np.flatnonzero((slots < available) & (share > 0))
        if len(room) == 0:
            break
        deficit = share[room] - slots[room]
        slots[room[np.argmax(deficit)]] += 1
    return slots


def mmr(
    candidates: list[Scored], vectors: np.ndarray, k: int, lam: float, chosen: list[int]
) -> list[Scored]:
    """Pick k items by maximal marginal relevance against everything already chosen."""
    picked: list[Scored] = []
    pool = list(candidates)
    selected = list(chosen)
    while pool and len(picked) < k:
        if selected:
            redundancy = (vectors[[c.index for c in pool]] @ vectors[selected].T).max(axis=1)
        else:
            redundancy = np.zeros(len(pool))
        gains = [lam * c.score - (1 - lam) * r for c, r in zip(pool, redundancy, strict=True)]
        best = int(np.argmax(gains))
        picked.append(pool.pop(best))
        selected.append(picked[-1].index)
    return picked


def build_slate(
    scored: list[Scored],
    weights: np.ndarray,
    vectors: np.ndarray,
    feed_size: int,
    explore: list[int],
    explore_slots: int,
    rule: list[Scored],
    lam: float,
    max_rule: int = 3,
) -> list[Slot]:
    rule = sorted(rule, key=lambda s: -s.score)[:max_rule]
    taken = {s.index for s in rule}
    explore = [i for i in explore if i not in taken][:explore_slots]
    taken |= set(explore)
    n_main = feed_size - len(rule) - len(explore)

    by_interest: dict[int, list[Scored]] = {}
    for s in sorted(scored, key=lambda s: -s.score):
        if s.index not in taken:
            by_interest.setdefault(s.interest, []).append(s)
    available = np.array([len(by_interest.get(c, [])) for c in range(len(weights))])
    quota = allocate(weights, available, n_main)

    chosen = [s.index for s in rule]
    picks_by_interest: list[list[Scored]] = []
    for c in np.argsort(-weights, kind="stable"):
        picks = mmr(by_interest.get(int(c), []), vectors, int(quota[c]), lam, chosen)
        picks_by_interest.append(picks)
        chosen += [p.index for p in picks]

    # Rerank scores are not comparable across queries, so interleave interests instead.
    main = [
        picks[j]
        for j in range(max((len(p) for p in picks_by_interest), default=0))
        for picks in picks_by_interest
        if j < len(picks)
    ]
    slate = [Slot(s.index, s.interest, "rule") for s in rule]
    slate += [Slot(s.index, s.interest, "main") for s in main]
    # Exploration goes at roughly 1/3 and 2/3 of the list.
    for j, idx in enumerate(explore):
        pos = min(len(slate), (j + 1) * len(slate) // (len(explore) + 1) + j)
        slate.insert(pos, Slot(idx, None, "explore"))
    return slate
