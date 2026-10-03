"""Stage 1: cheap candidate generation over one day's papers."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Candidates:
    sims: np.ndarray  # (n_papers, n_interests) cosine similarity
    per_interest: list[np.ndarray]  # paper indices, best first, top-M per interest
    explore: np.ndarray  # paper indices sampled from the near-miss band


def generate(
    papers: np.ndarray,
    interests: np.ndarray,
    top_m: int,
    band: tuple[int, int],
    pool_size: int,
    rng: np.random.Generator,
) -> Candidates:
    """Top-M papers per interest by cosine similarity, plus an exploration pool."""
    sims = papers @ interests.T
    order = np.argsort(-sims, axis=0)  # per interest, paper indices best first
    per_interest = [order[:top_m, c] for c in range(interests.shape[0])]

    # Best (lowest) rank of each paper across interests.
    ranks = np.empty_like(order)
    for c in range(order.shape[1]):
        ranks[order[:, c], c] = np.arange(order.shape[0])
    best_rank = ranks.min(axis=1)
    in_band = np.flatnonzero((best_rank >= band[0]) & (best_rank < band[1]))
    explore = rng.choice(in_band, size=min(pool_size, len(in_band)), replace=False)
    return Candidates(sims=sims, per_interest=per_interest, explore=np.sort(explore))
