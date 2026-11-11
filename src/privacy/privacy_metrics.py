"""Empirical edge-reconstruction and utility indicators for released graphs.

IMPORTANT: precision, recall, F1, Jaccard, retention and density are **empirical indicators**
computed against the known original graph. They are NOT formal differential-privacy metrics. The
formal guarantee comes only from the mechanism (see :mod:`src.privacy.differential_privacy`).

All functions take 0/1 pair vectors (see :mod:`src.graph_generation.graph_preprocessing`).
Convention: when a denominator is 0 the metric is defined as 0.0 (documented per function).
"""
from __future__ import annotations

import numpy as np

from src.graph_generation.randomized_response import flip_probability


def _check(orig: np.ndarray, rel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    orig, rel = np.asarray(orig).astype(bool), np.asarray(rel).astype(bool)
    if orig.shape != rel.shape or orig.ndim != 1:
        raise ValueError("orig and rel must be 1-D vectors of the same length.")
    return orig, rel


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def edge_precision(orig: np.ndarray, rel: np.ndarray) -> float:
    """|E_G ∩ E_H| / |E_H|. Higher = a released edge is more likely real (higher disclosure).
    Equals the adversary's posterior belief that an observed edge is real (empirical). 0 if |E_H|=0."""
    o, r = _check(orig, rel)
    return _safe_div((o & r).sum(), r.sum())


def edge_recall(orig: np.ndarray, rel: np.ndarray) -> float:
    """|E_G ∩ E_H| / |E_G|. Fraction of real edges present in the release. 0 if |E_G|=0."""
    o, r = _check(orig, rel)
    return _safe_div((o & r).sum(), o.sum())


def edge_f1(orig: np.ndarray, rel: np.ndarray) -> float:
    """Harmonic mean of precision and recall (= 2|∩| / (|E_G| + |E_H|)). 0 if both empty."""
    o, r = _check(orig, rel)
    return _safe_div(2.0 * (o & r).sum(), o.sum() + r.sum())


def edge_jaccard(orig: np.ndarray, rel: np.ndarray) -> float:
    """|E_G ∩ E_H| / |E_G ∪ E_H|. Edge overlap; 1 = identical edge sets. 0 if both empty."""
    o, r = _check(orig, rel)
    return _safe_div((o & r).sum(), (o | r).sum())


def edge_retention(orig: np.ndarray, rel: np.ndarray) -> float:
    """Fraction of ORIGINAL edges that survive in the release (numerically equal to recall;
    kept as a separate name for readability in tables/plots)."""
    return edge_recall(orig, rel)


def edge_count_ratio(orig: np.ndarray, rel: np.ndarray) -> float:
    """|E_H| / |E_G|. Values >> 1 mean the release is dominated by fake edges. 0 if |E_G|=0."""
    o, r = _check(orig, rel)
    return _safe_div(r.sum(), o.sum())


def graph_density(vec: np.ndarray) -> float:
    """Fraction of node pairs that are edges, |E| / N."""
    v = np.asarray(vec).astype(bool)
    return _safe_div(v.sum(), v.size)


def flip_fraction(orig: np.ndarray, rel: np.ndarray) -> float:
    """Empirical fraction of pairs whose bit differs; estimates the flip probability p."""
    o, r = _check(orig, rel)
    return float((o != r).mean())


def edges_added(orig: np.ndarray, rel: np.ndarray) -> int:
    """Number of fake edges (pairs that are non-edges in G but edges in H)."""
    o, r = _check(orig, rel)
    return int((~o & r).sum())


def edges_deleted(orig: np.ndarray, rel: np.ndarray) -> int:
    """Number of real edges missing from H."""
    o, r = _check(orig, rel)
    return int((o & ~r).sum())


def edge_inference_advantage(prior: float, posterior: float) -> float:
    """Posterior belief minus prior for an observed edge (empirical/analytic disclosure indicator).
    0 = the observation taught the adversary nothing; larger = more disclosure. Prior-dependent."""
    return float(posterior - prior)


def all_edge_metrics(orig: np.ndarray, rel: np.ndarray) -> dict[str, float]:
    """All empirical edge indicators for one release."""
    return {
        "flip_fraction": flip_fraction(orig, rel),
        "precision": edge_precision(orig, rel),
        "recall": edge_recall(orig, rel),
        "f1": edge_f1(orig, rel),
        "jaccard": edge_jaccard(orig, rel),
        "retention": edge_retention(orig, rel),
        "edge_count_ratio": edge_count_ratio(orig, rel),
        "density": graph_density(rel),
        "edges_added": edges_added(orig, rel),
        "edges_deleted": edges_deleted(orig, rel),
        "n_edges": int(np.asarray(rel).sum()),
    }


def analytic_rr_metrics(n_pairs: int, n_edges: int, epsilon: float) -> dict[str, float]:
    """Plug-in (ratio-of-expectations) values of the indicators under RR.

    With p the flip probability, m real edges, N pairs: E|kept| = m(1-p), E|added| = (N-m)p.
    These are first-order approximations for the ratio metrics (precision, F1, Jaccard); they
    converge to the empirical mean as N grows.
    """
    p = flip_probability(epsilon)
    m, N = n_edges, n_pairs
    kept, added = m * (1 - p), (N - m) * p
    total = kept + added
    return {
        "flip_probability": p,
        "precision": _safe_div(kept, total),
        "recall": 1 - p if m > 0 else 0.0,
        "f1": _safe_div(2 * kept, m + total),
        "jaccard": _safe_div(kept, m + added),
        "retention": 1 - p if m > 0 else 0.0,
        "edge_count_ratio": _safe_div(total, m),
        "density": total / N if N > 0 else 0.0,
    }
