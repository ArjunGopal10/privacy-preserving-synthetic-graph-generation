# Project overview

**Privacy-Preserving Synthetic Graph Generation Using Multi-Objective Optimization** — an M.Tech AI project by Arjun Gopal.

## Research question
How much community structure of a small social network survives when the graph is released with
*edge-level differential privacy*, and what does the privacy–utility trade-off curve look like?

## Pipeline
```
original graph G (Karate Club) ──► fixed reference partition P0 (Louvain, computed once)
        │
        ├─► randomized response at epsilon ──► released graph H ──► privacy indicators (empirical)
        │                                                      └─► community metrics (fixed P0 / re-detected)
        └─► NSGA-II over epsilon: min [ epsilon , -E[Q(H; P0)] ]  ──► Pareto front, compared with a dense grid
```
Baselines (matched random graph, degree-preserving rewiring, deletion-only, Laplace top-m filter) are
evaluated with the same metrics (`experiments/run_baseline_comparison.py`).

## What is and is not claimed
| Claim | Status |
|---|---|
| One randomized-response release at fixed epsilon is epsilon-edge-DP | Yes (formal, see `differential_privacy.md`) |
| Node-level DP | **No** |
| Identity information is anonymised | **No** — DP bounds what a release reveals about one *edge*; it is not a statement about names or auxiliary information |
| The NSGA-II + candidate-selection pipeline is end-to-end DP | **No** — candidates are scored against the original graph |
| NSGA-II finds a globally optimal epsilon | **No** — epsilon is the only variable; the Pareto set is a curve, and a dense grid is an equally good (and simpler) reference |

## Where things are
* Code: `src/` (modules), `experiments/` (runnable scripts), `tests/`.
* Results: `results/` (CSV tables, figures with underlying data, JSON metadata/manifest).
* Explanations: this `docs/` folder. Interactive explainer: https://arjungopalevolveproject.netlify.app/
