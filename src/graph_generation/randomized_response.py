"""Randomized Response (RR) for edge-level differential privacy.

Every one of the N node pairs is flipped independently with probability

    p = 1 / (1 + e^epsilon).

Neighbouring graphs differ in a single pair (edge adjacency); for that pair the output
distribution ratio is (1 - p) / p = e^epsilon, so one release is epsilon-edge-DP.

Common random numbers (CRN)
---------------------------
All releases are produced as ``flip = (U < p)`` with a pre-drawn uniform matrix ``U``.
Re-using the same ``U`` for different epsilon values gives *coupled* releases: because p decreases
with epsilon, the set of flipped pairs at a larger epsilon is a subset of the set at a smaller one.
This removes sampling noise from epsilon-to-epsilon comparisons without changing the marginal
distribution of any individual release.
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
from scipy.special import expit

from .graph_preprocessing import graph_to_pair_vector, num_possible_pairs, pair_vector_to_graph


def validate_epsilon(epsilon: float) -> float:
    """Return ``epsilon`` as float; allow 0 (p = 0.5) and +inf (p = 0), reject negative/NaN."""
    eps = float(epsilon)
    if math.isnan(eps) or eps < 0:
        raise ValueError(f"epsilon must be >= 0 (got {epsilon!r}).")
    return eps


def flip_probability(epsilon: float) -> float:
    """RR flip probability ``p = 1 / (1 + e^epsilon)`` (numerically stable for large epsilon)."""
    return float(expit(-validate_epsilon(epsilon)))


def draw_uniforms(n_pairs: int, n_releases: int, seed: int | None) -> np.ndarray:
    """Matrix of shape ``(n_releases, n_pairs)`` of U(0,1) draws (the CRN source)."""
    if n_pairs < 0 or n_releases < 1:
        raise ValueError("n_pairs must be >= 0 and n_releases >= 1.")
    return np.random.default_rng(seed).random((n_releases, n_pairs))


def randomized_response_vectors(vec: np.ndarray, epsilon: float, uniforms: np.ndarray) -> np.ndarray:
    """Apply RR to a pair vector for each row of ``uniforms``; returns ``(K, N)`` uint8."""
    vec = np.asarray(vec, dtype=np.uint8)
    uniforms = np.atleast_2d(uniforms)
    if uniforms.shape[1] != vec.shape[0]:
        raise ValueError("uniforms must have one column per node pair.")
    flips = (uniforms < flip_probability(epsilon)).astype(np.uint8)
    return np.bitwise_xor(vec[None, :], flips)


def randomized_response_graph(G: nx.Graph, epsilon: float,
                              rng: np.random.Generator | int | None = None) -> nx.Graph:
    """One RR release of ``G`` as a NetworkX graph (same node set ``0..n-1``)."""
    n = G.number_of_nodes()
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    u = gen.random((1, num_possible_pairs(n)))
    out = randomized_response_vectors(graph_to_pair_vector(G), epsilon, u)[0]
    return pair_vector_to_graph(out, n)
