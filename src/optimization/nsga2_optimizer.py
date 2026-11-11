"""NSGA-II driver: pymoo run with generation-wise tracking, a from-scratch cross-check run, and the
dense grid-search baseline.

Because epsilon is the ONLY decision variable, the search space is one-dimensional. A dense grid
is therefore an (almost) exact reference; NSGA-II is expected to match it, not beat it. NSGA-II does
not "discover a globally optimal epsilon": the Pareto set is a *family* of trade-off solutions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

from .fitness_functions import ExpectedModularityEvaluator
from .nsga2_components import assign_rank_and_crowding, run_nsga2_from_scratch
from .nsga2_problem import EpsilonPrivacyUtilityProblem
from .pareto_analysis import extract_pareto_front, hypervolume_2d, non_dominated_mask


@dataclass
class NSGA2Config:
    """All NSGA-II settings (defaults reproduce the project's existing configuration)."""
    pop_size: int = 40
    n_gen: int = 30
    seed: int = 42
    crossover_prob: float = 0.9
    crossover_eta: float = 15.0
    mutation_eta: float = 20.0
    eps_min: float = 0.1
    eps_max: float = 10.0
    n_releases: int = 20
    common_random_numbers: bool = True
    hv_ref_point: tuple[float, float] = (10.0, 0.1)

    def __post_init__(self) -> None:
        if self.pop_size < 4 or self.n_gen < 1 or self.n_releases < 1:
            raise ValueError("Need pop_size >= 4, n_gen >= 1 and n_releases >= 1.")
        if not 0 <= self.eps_min < self.eps_max:
            raise ValueError("Need 0 <= eps_min < eps_max.")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NSGA2Result:
    """Outputs of one optimisation run."""
    front: pd.DataFrame               # unique non-dominated solutions of the FINAL population
    generations: pd.DataFrame         # one row per generation (diagnostics)
    populations: pd.DataFrame         # every individual of every generation
    final_X: np.ndarray
    final_F: np.ndarray
    n_evaluations: int
    config: NSGA2Config
    backend: str


def _generation_row(gen: int, X: np.ndarray, F: np.ndarray, rank: np.ndarray, crowd: np.ndarray,
                    cfg: NSGA2Config, n_evals: int) -> dict:
    finite = crowd[np.isfinite(crowd) & (rank == 0)]
    return {
        "generation": gen, "n_evaluations": n_evals,
        "front_size": int((rank == 0).sum()),
        "front_size_unique": int(len(np.unique(np.round(F[rank == 0], 10), axis=0))),
        "hypervolume": hypervolume_2d(F, cfg.hv_ref_point),
        "best_expected_q": float(-F[:, 1].min()),
        "min_epsilon": float(X.min()), "mean_epsilon": float(X.mean()), "max_epsilon": float(X.max()),
        "epsilon_std": float(X.std()),
        "mean_finite_crowding_front0": float(finite.mean()) if finite.size else float("nan"),
    }


def _population_rows(gen: int, X: np.ndarray, F: np.ndarray, rank: np.ndarray, crowd: np.ndarray):
    return [{"generation": gen, "individual": i, "epsilon": float(X[i, 0]),
             "neg_expected_q": float(F[i, 1]), "expected_q": float(-F[i, 1]),
             "rank": int(rank[i]), "crowding_distance": float(crowd[i])} for i in range(len(X))]


class _HistoryCallback(Callback):
    def __init__(self, cfg: NSGA2Config, evaluator: ExpectedModularityEvaluator):
        super().__init__()
        self.cfg, self.evaluator = cfg, evaluator
        self.gen_rows: list[dict] = []
        self.pop_rows: list[dict] = []

    def notify(self, algorithm):
        X, F = algorithm.pop.get("X"), algorithm.pop.get("F")
        rank, crowd = assign_rank_and_crowding(F)
        gen = int(algorithm.n_gen)
        self.gen_rows.append(_generation_row(gen, X, F, rank, crowd, self.cfg, algorithm.evaluator.n_eval))
        self.pop_rows.extend(_population_rows(gen, X, F, rank, crowd))


def run_nsga2(evaluator: ExpectedModularityEvaluator, cfg: NSGA2Config | None = None) -> NSGA2Result:
    """Run pymoo's NSGA-II on the (epsilon, -E[Q]) problem and track every generation."""
    cfg = cfg or NSGA2Config()
    problem = EpsilonPrivacyUtilityProblem(evaluator, cfg.eps_min, cfg.eps_max)
    algorithm = NSGA2(
        pop_size=cfg.pop_size,
        crossover=SBX(prob=cfg.crossover_prob, eta=cfg.crossover_eta),
        mutation=PM(eta=cfg.mutation_eta),
        eliminate_duplicates=True,
    )
    cb = _HistoryCallback(cfg, evaluator)
    res = minimize(problem, algorithm, ("n_gen", cfg.n_gen), seed=cfg.seed, callback=cb, verbose=False)
    X, F = res.pop.get("X"), res.pop.get("F")
    return NSGA2Result(
        front=extract_pareto_front(X, F), generations=pd.DataFrame(cb.gen_rows),
        populations=pd.DataFrame(cb.pop_rows), final_X=X, final_F=F,
        n_evaluations=int(res.algorithm.evaluator.n_eval), config=cfg, backend="pymoo")


def run_nsga2_scratch(evaluator: ExpectedModularityEvaluator, cfg: NSGA2Config | None = None
                      ) -> NSGA2Result:
    """Same problem solved with the from-scratch components (cross-check of the pymoo run)."""
    cfg = cfg or NSGA2Config()
    hist = run_nsga2_from_scratch(
        lambda x: evaluator.objectives(float(x[0])), np.array([cfg.eps_min]), np.array([cfg.eps_max]), cfg.pop_size,
        cfg.n_gen, cfg.seed, cfg.crossover_prob, cfg.crossover_eta, cfg.mutation_eta)
    gen_rows, pop_rows = [], []
    for h in hist:
        rank, crowd = assign_rank_and_crowding(h["F"])
        n_evals = cfg.pop_size + (h["gen"] - 1) * (cfg.pop_size // 2 * 2)
        gen_rows.append(_generation_row(h["gen"], h["X"], h["F"], rank, crowd, cfg, n_evals))
        pop_rows.extend(_population_rows(h["gen"], h["X"], h["F"], rank, crowd))
    X, F = hist[-1]["X"], hist[-1]["F"]
    return NSGA2Result(extract_pareto_front(X, F), pd.DataFrame(gen_rows), pd.DataFrame(pop_rows),
                       X, F, evaluator.n_evaluations, cfg, "from_scratch")


def grid_search(evaluator: ExpectedModularityEvaluator, eps_min: float = 0.1, eps_max: float = 10.0,
                n_points: int = 500) -> pd.DataFrame:
    """Dense grid baseline: E[Q] (and SD over releases) at ``n_points`` equally spaced epsilons.

    Costs ``n_points`` evaluations versus pop_size * n_gen for NSGA-II; with a single decision
    variable the grid is the stronger, simpler baseline.
    """
    grid = np.linspace(eps_min, eps_max, n_points)
    rows = []
    for e in grid:
        q = evaluator.modularity_samples(float(e))
        rows.append({"epsilon": float(e), "expected_q": float(q.mean()), "sd_q": float(q.std(ddof=1)
                     if q.size > 1 else 0.0)})
    df = pd.DataFrame(rows)
    df["neg_expected_q"] = -df["expected_q"]
    df["on_pareto_front"] = non_dominated_mask(df[["epsilon", "neg_expected_q"]].to_numpy())
    return df
