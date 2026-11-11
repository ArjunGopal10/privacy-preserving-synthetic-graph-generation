"""Reference NumPy implementation of the NSGA-II building blocks (Deb et al., 2002).

These are used (a) to document/teach the algorithm, (b) for diversity diagnostics, and (c) to run
a from-scratch NSGA-II that is cross-checked against the pymoo run. All objectives are minimised.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def fast_non_dominated_sort(F: np.ndarray) -> list[np.ndarray]:
    """Partition row indices of ``F`` into Pareto fronts (front 0 = non-dominated)."""
    F = np.asarray(F, float)
    n = F.shape[0]
    dominated_by = [[] for _ in range(n)]        # S_p: solutions dominated by p
    dom_count = np.zeros(n, dtype=int)           # n_p: number of solutions dominating p
    for p in range(n):
        le = np.all(F[p] <= F, axis=1)
        lt = np.any(F[p] < F, axis=1)
        dominated_by[p] = list(np.flatnonzero(le & lt))
        dom_count[p] = int(np.sum(np.all(F <= F[p], axis=1) & np.any(F < F[p], axis=1)))
    fronts: list[np.ndarray] = []
    current = np.flatnonzero(dom_count == 0)
    while current.size:
        fronts.append(current)
        nxt = []
        for p in current:
            for q in dominated_by[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    nxt.append(q)
        current = np.array(sorted(nxt), dtype=int)
    return fronts


def ranks_from_fronts(fronts: list[np.ndarray], n: int) -> np.ndarray:
    rank = np.zeros(n, dtype=int)
    for r, idx in enumerate(fronts):
        rank[idx] = r
    return rank


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """Crowding distance of each row within ONE front (boundary points get ``inf``)."""
    F = np.asarray(F, float)
    n, m = F.shape
    dist = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        order = np.argsort(F[:, j], kind="stable")
        span = F[order[-1], j] - F[order[0], j]
        dist[order[0]] = dist[order[-1]] = np.inf
        if span <= 0:
            continue
        dist[order[1:-1]] += (F[order[2:], j] - F[order[:-2], j]) / span
    return dist


def assign_rank_and_crowding(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank and (per-front) crowding distance for every row of ``F``."""
    fronts = fast_non_dominated_sort(F)
    rank = ranks_from_fronts(fronts, len(F))
    crowd = np.zeros(len(F))
    for idx in fronts:
        crowd[idx] = crowding_distance(F[idx])
    return rank, crowd


def binary_tournament_selection(rank: np.ndarray, crowd: np.ndarray, n_select: int,
                                rng: np.random.Generator) -> np.ndarray:
    """Crowded-comparison binary tournament: lower rank wins; ties -> larger crowding distance."""
    n = len(rank)
    a, b = rng.integers(0, n, n_select), rng.integers(0, n, n_select)
    a_wins = (rank[a] < rank[b]) | ((rank[a] == rank[b]) & (crowd[a] > crowd[b]))
    tie = (rank[a] == rank[b]) & (crowd[a] == crowd[b])
    a_wins = np.where(tie, rng.random(n_select) < 0.5, a_wins)
    return np.where(a_wins, a, b)


def _sbx_betaq(beta: np.ndarray, u: np.ndarray, eta: float) -> np.ndarray:
    alpha = 2.0 - beta ** (-(eta + 1.0))
    return np.where(u <= 1.0 / alpha, (u * alpha) ** (1.0 / (eta + 1.0)),
                    (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta + 1.0)))


def sbx_crossover(p1: np.ndarray, p2: np.ndarray, xl: np.ndarray, xu: np.ndarray, eta: float = 15.0,
                  prob: float = 0.9, prob_var: float = 0.5, rng: np.random.Generator | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Bounded simulated binary crossover on arrays of shape ``(n_pairs, n_var)``."""
    rng = rng or np.random.default_rng()
    p1, p2 = np.atleast_2d(p1).astype(float), np.atleast_2d(p2).astype(float)
    c1, c2 = p1.copy(), p2.copy()
    n, d = p1.shape
    do_pair = (rng.random(n) < prob)[:, None]
    do_var = rng.random((n, d)) < prob_var
    differ = np.abs(p1 - p2) > 1e-14
    active = do_pair & do_var & differ
    y1, y2 = np.minimum(p1, p2), np.maximum(p1, p2)
    safe = np.where(active, y2 - y1, 1.0)
    u = rng.random((n, d))
    bq1 = _sbx_betaq(1.0 + 2.0 * (y1 - xl) / safe, u, eta)
    bq2 = _sbx_betaq(1.0 + 2.0 * (xu - y2) / safe, u, eta)
    o1 = np.clip(0.5 * ((y1 + y2) - bq1 * (y2 - y1)), xl, xu)
    o2 = np.clip(0.5 * ((y1 + y2) + bq2 * (y2 - y1)), xl, xu)
    swap = rng.random((n, d)) < 0.5
    c1 = np.where(active, np.where(swap, o2, o1), c1)
    c2 = np.where(active, np.where(swap, o1, o2), c2)
    return c1, c2


def polynomial_mutation(X: np.ndarray, xl: np.ndarray, xu: np.ndarray, eta: float = 20.0,
                        prob_var: float | None = None, rng: np.random.Generator | None = None
                        ) -> np.ndarray:
    """Bounded polynomial mutation. Default per-variable probability is ``min(0.5, 1/n_var)``."""
    rng = rng or np.random.default_rng()
    X = np.atleast_2d(X).astype(float)
    n, d = X.shape
    pm = min(0.5, 1.0 / d) if prob_var is None else prob_var
    rngw = xu - xl
    d1, d2 = (X - xl) / rngw, (xu - X) / rngw
    u = rng.random((n, d))
    pw = 1.0 / (eta + 1.0)
    low = 2 * u + (1 - 2 * u) * (1 - d1) ** (eta + 1)
    high = 2 * (1 - u) + 2 * (u - 0.5) * (1 - d2) ** (eta + 1)
    dq = np.where(u < 0.5, low ** pw - 1.0, 1.0 - high ** pw)
    mutate = rng.random((n, d)) < pm
    return np.clip(np.where(mutate, X + dq * rngw, X), xl, xu)


def environmental_selection(F: np.ndarray, n_survive: int) -> np.ndarray:
    """Indices of the ``n_survive`` best rows: fill whole fronts in order, split the last front by
    descending crowding distance."""
    chosen: list[int] = []
    for idx in fast_non_dominated_sort(F):
        if len(chosen) + len(idx) <= n_survive:
            chosen.extend(idx.tolist())
        else:
            cd = crowding_distance(F[idx])
            take = idx[np.argsort(-cd, kind="stable")[: n_survive - len(chosen)]]
            chosen.extend(take.tolist())
            break
        if len(chosen) == n_survive:
            break
    return np.array(chosen, dtype=int)


def run_nsga2_from_scratch(evaluate: Callable[[np.ndarray], np.ndarray], xl: np.ndarray, xu: np.ndarray,
                           pop_size: int = 40, n_gen: int = 30, seed: int = 42,
                           crossover_prob: float = 0.9, crossover_eta: float = 15.0,
                           mutation_eta: float = 20.0) -> list[dict]:
    """Textbook NSGA-II loop. ``evaluate`` maps one decision vector to an objective vector.

    Returns one dict per generation with keys ``gen, X, F`` (the surviving population).
    """
    rng = np.random.default_rng(seed)
    xl, xu = np.atleast_1d(xl).astype(float), np.atleast_1d(xu).astype(float)
    X = xl + rng.random((pop_size, xl.size)) * (xu - xl)            # 1. initialisation
    F = np.vstack([evaluate(x) for x in X])                          # 2. evaluation
    history = [{"gen": 1, "X": X.copy(), "F": F.copy()}]
    for gen in range(2, n_gen + 1):
        rank, crowd = assign_rank_and_crowding(F)
        parents = binary_tournament_selection(rank, crowd, pop_size, rng)         # 5. selection
        half = pop_size // 2 * 2
        pa, pb = X[parents[:half:2]], X[parents[1:half:2]]
        ca, cb = sbx_crossover(pa, pb, xl, xu, crossover_eta, crossover_prob, rng=rng)   # 6. SBX
        children = polynomial_mutation(np.vstack([ca, cb]), xl, xu, mutation_eta, rng=rng)  # 7. PM
        Fc = np.vstack([evaluate(x) for x in children])
        Xall, Fall = np.vstack([X, children]), np.vstack([F, Fc])
        keep = environmental_selection(Fall, pop_size)               # 8. elitist survival
        X, F = Xall[keep], Fall[keep]
        history.append({"gen": gen, "X": X.copy(), "F": F.copy()})
    return history
