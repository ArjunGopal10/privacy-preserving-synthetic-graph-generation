"""Pareto-front extraction, hypervolume, solution selection and NSGA-II-vs-grid comparison.

All objectives are minimised: columns of ``F`` are ``[epsilon, -E[Q]]``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def non_dominated_mask(F: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows (minimisation). Exact duplicates are all kept."""
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        dominated = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        if dominated.any():
            mask[i] = False
    return mask


def extract_pareto_front(X: np.ndarray, F: np.ndarray, decimals: int = 10) -> pd.DataFrame:
    """Unique non-dominated solutions sorted by epsilon; columns epsilon, neg_expected_q, expected_q."""
    X, F = np.asarray(X, float).reshape(len(F), -1), np.asarray(F, float)
    keep = non_dominated_mask(F)
    df = pd.DataFrame({"epsilon": X[keep, 0], "neg_expected_q": F[keep, 1]})
    df = df.round(decimals).drop_duplicates().sort_values("epsilon").reset_index(drop=True)
    df["expected_q"] = -df["neg_expected_q"]
    return df


def hypervolume_2d(F: np.ndarray, ref_point: tuple[float, float]) -> float:
    """Area dominated by the non-dominated subset of ``F`` and bounded by ``ref_point`` (min.)."""
    F = np.asarray(F, float)
    F = F[np.all(F < np.asarray(ref_point), axis=1)]
    if F.size == 0:
        return 0.0
    F = F[non_dominated_mask(F)]
    F = F[np.argsort(F[:, 0])]
    hv, prev_f2 = 0.0, ref_point[1]
    for f1, f2 in F:
        if f2 < prev_f2:
            hv += (ref_point[0] - f1) * (prev_f2 - f2)
            prev_f2 = f2
    return float(hv)


def select_solutions(front: pd.DataFrame, original_q: float,
                     utility_fractions: tuple[float, ...] = (0.5, 0.8, 0.9)) -> pd.DataFrame:
    """Pick named operating points from a front (smallest epsilon reaching a fraction of Q0, and
    the knee = point farthest from the line joining the front's extremes in normalised space).

    Selecting a point by comparing with the ORIGINAL graph's modularity is data-dependent.
    """
    f = front.sort_values("epsilon").reset_index(drop=True)
    rows = []
    for frac in utility_fractions:
        ok = f[f["expected_q"] >= frac * original_q]
        if not ok.empty:
            r = ok.iloc[0]
            rows.append({"label": f"smallest eps with E[Q] >= {int(frac * 100)}% of Q0",
                         "epsilon": r.epsilon, "expected_q": r.expected_q,
                         "utility_fraction": r.expected_q / original_q})
    if len(f) >= 3:
        e = (f["epsilon"] - f["epsilon"].min()) / max(np.ptp(f["epsilon"]), 1e-12)
        q = (f["expected_q"] - f["expected_q"].min()) / max(np.ptp(f["expected_q"]), 1e-12)
        d = np.abs(q - e) / np.sqrt(2)           # distance to the line q = e (diagonal)
        r = f.iloc[int(np.argmax(d.to_numpy()))]
        rows.append({"label": "knee (max distance from diagonal)", "epsilon": r.epsilon,
                     "expected_q": r.expected_q, "utility_fraction": r.expected_q / original_q})
    return pd.DataFrame(rows)


def compare_with_grid(nsga_front: pd.DataFrame, grid_df: pd.DataFrame,
                      ref_point: tuple[float, float]) -> dict[str, float]:
    """Quality of an NSGA-II front relative to the dense-grid front (the near-exact reference).

    ``hv_ratio`` = HV(NSGA-II) / HV(grid). ``max_q_gap`` = largest shortfall of NSGA-II's
    expected modularity versus the grid envelope at the same epsilon (>= 0 means worse).
    """
    gF = np.column_stack([grid_df["epsilon"], -grid_df["expected_q"]])
    nF = np.column_stack([nsga_front["epsilon"], nsga_front["neg_expected_q"]])
    grid_front = gF[non_dominated_mask(gF)]
    hv_g, hv_n = hypervolume_2d(grid_front, ref_point), hypervolume_2d(nF, ref_point)
    order = np.argsort(grid_df["epsilon"].to_numpy())
    q_env = np.maximum.accumulate(grid_df["expected_q"].to_numpy()[order])
    q_at = np.interp(nsga_front["epsilon"], grid_df["epsilon"].to_numpy()[order], q_env)
    gaps = q_at - nsga_front["expected_q"].to_numpy()
    return {"hv_grid": hv_g, "hv_nsga2": hv_n, "hv_ratio": hv_n / hv_g if hv_g > 0 else float("nan"),
            "n_front_grid": int(len(grid_front)), "n_front_nsga2": int(len(nsga_front)),
            "max_q_gap": float(np.max(gaps)), "mean_q_gap": float(np.mean(gaps))}
