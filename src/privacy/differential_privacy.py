"""Formal differential-privacy quantities for edge-level randomized response.

Everything in this module is a *formal* (mechanism-level) statement, in contrast to the empirical
reconstruction indicators in :mod:`src.privacy.privacy_metrics`.

Adjacency: two graphs on the same node set are *edge-neighbours* if they differ in exactly one
node pair. This is **not** node-level DP.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from src.graph_generation.randomized_response import flip_probability, validate_epsilon


def epsilon_from_flip_probability(p: float) -> float:
    """Invert ``p = 1/(1+e^eps)``: ``eps = ln((1 - p) / p)``; requires ``0 < p <= 0.5``."""
    if not (0.0 < p <= 0.5):
        raise ValueError("flip probability must satisfy 0 < p <= 0.5.")
    return math.log((1.0 - p) / p)


def rr_output_matrix(epsilon: float) -> np.ndarray:
    """2x2 matrix ``M[input_bit, output_bit] = P(output | input)`` for one pair."""
    p = flip_probability(epsilon)
    return np.array([[1 - p, p], [p, 1 - p]])


def max_log_likelihood_ratio(epsilon: float) -> float:
    """Largest ``ln P(o | x) / P(o | x')`` over outputs ``o`` and inputs ``x != x'``.

    For RR this equals epsilon exactly (that is the epsilon-DP guarantee for one pair).
    """
    M = rr_output_matrix(epsilon)
    ratios = [math.log(M[0, o] / M[1, o]) for o in (0, 1) if M[0, o] > 0 and M[1, o] > 0]
    ratios += [math.log(M[1, o] / M[0, o]) for o in (0, 1) if M[0, o] > 0 and M[1, o] > 0]
    return max(ratios) if ratios else math.inf


def likelihood_ratio_bound(epsilon: float) -> float:
    """The multiplicative DP bound ``e^epsilon`` (``inf`` if it overflows)."""
    try:
        return math.exp(validate_epsilon(epsilon))
    except OverflowError:
        return math.inf


def basic_composition(epsilons: Iterable[float]) -> float:
    """Sequential (basic) composition: total budget = sum of the per-release budgets."""
    return float(sum(validate_epsilon(e) for e in epsilons))


def expected_release_counts(n_pairs: int, n_edges: int, epsilon: float) -> dict[str, float]:
    """Expected kept / deleted / added / total edges in one RR release."""
    if not 0 <= n_edges <= n_pairs:
        raise ValueError("Need 0 <= n_edges <= n_pairs.")
    p = flip_probability(epsilon)
    kept = n_edges * (1 - p)
    added = (n_pairs - n_edges) * p
    return {"kept": kept, "deleted": n_edges * p, "added": added, "total_edges": kept + added}


def posterior_edge_belief(prior: float, epsilon: float, observed_edge: bool = True) -> float:
    """Bayesian posterior that a pair is a real edge after seeing one RR output bit.

    ``prior`` is the adversary's prior edge probability (e.g. graph density). This is an
    *interpretation* of the formal guarantee under a specific prior, not an additional guarantee.
    """
    if not 0.0 <= prior <= 1.0:
        raise ValueError("prior must be in [0, 1].")
    p = flip_probability(epsilon)
    like_real, like_fake = (1 - p, p) if observed_edge else (p, 1 - p)
    denom = like_real * prior + like_fake * (1 - prior)
    return float(like_real * prior / denom) if denom > 0 else float("nan")
