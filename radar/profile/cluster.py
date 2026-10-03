"""Turn a set of profile-paper vectors into weighted interests."""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def recency_weights(ages_days: np.ndarray, tau_days: float) -> np.ndarray:
    """w_i = exp(-a_i / tau): a paper from a year ago counts ~37% as much as today's."""
    return np.exp(-np.clip(ages_days, 0, None) / tau_days)


@dataclass
class Clustering:
    labels: np.ndarray  # cluster index per paper
    medoids: list[int]  # index of the medoid paper, per cluster
    mass: np.ndarray  # sum of recency weights, per cluster


# 99th percentile cosine of random same-day paper pairs (median 0.22), measured on Embed 4.
SAME_TOPIC_COSINE = 0.40


def small_profile_labels(x: np.ndarray, threshold: float = SAME_TOPIC_COSINE) -> np.ndarray:
    """Connected components of the "same topic" graph."""
    n = len(x)
    labels = np.arange(n)
    sims = x @ x.T
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= threshold:
                labels[labels == labels[j]] = labels[i]
    _, labels = np.unique(labels, return_inverse=True)
    return labels


def choose_clusters(
    x: np.ndarray, w: np.ndarray, k_max: int, min_cluster_size: int = 2, seed: int = 0
) -> Clustering:
    """Weighted k-means for k = 2..k_max, choosing k by cosine silhouette."""
    n = len(x)
    best_labels = np.zeros(n, dtype=int)
    best_score = -np.inf
    if n < 4:
        best_labels = small_profile_labels(x)
        while best_labels.max() + 1 > k_max:
            best_labels[best_labels == best_labels.max()] = 0
    else:
        for k in range(2, min(k_max, n - 1) + 1):
            labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(
                x, sample_weight=w
            )
            if np.bincount(labels, minlength=k).min() < min_cluster_size:
                continue
            score = silhouette_score(x, labels, metric="cosine")
            if score > best_score:
                best_score, best_labels = score, labels
    return Clustering(
        labels=best_labels,
        medoids=[medoid(x, w, best_labels == c) for c in range(best_labels.max() + 1)],
        mass=np.array([w[best_labels == c].sum() for c in range(best_labels.max() + 1)]),
    )


def medoid(x: np.ndarray, w: np.ndarray, members: np.ndarray) -> int:
    """argmax_j sum_i w_i s(e_i, e_j) over members: the real paper most central to the cluster."""
    idx = np.flatnonzero(members)
    sims = x[idx] @ x[idx].T
    return int(idx[np.argmax(sims.T @ w[idx])])


def normalize_weights(mass: np.ndarray) -> np.ndarray:
    total = mass.sum()
    return mass / total if total > 0 else np.full(len(mass), 1 / max(len(mass), 1))
