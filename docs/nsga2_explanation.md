# NSGA-II in this project

## Problem
Minimise $[f_1,f_2]=[\varepsilon,\,-\mathbb E[Q(H_\varepsilon;P_0)]]$ over $\varepsilon\in[0.1,10]$.

**Why is modularity negated?** NSGA-II (and pymoo) *minimise* every objective. We want high modularity, so we minimise its negative; $\min(-Q)\equiv\max Q$. Figures plot $\mathbb E[Q]$ (= $-f_2$) so "up" means better utility.

**Why two objectives?** Privacy loss and utility conflict; a weighted sum $(1-w)\varepsilon_{\text{norm}}-wQ/Q_0$ returns one point whose location depends on the arbitrary weight $w$. NSGA-II returns the whole non-dominated set so the trade-off can be inspected (a weighted-sum helper is provided as `combined_fitness` for comparison only).

## Algorithm (Deb et al., 2002) and where it lives
| Step | Description | Code |
|---|---|---|
| 1 Initialise | uniform random $\varepsilon$ in bounds | pymoo `FloatRandomSampling`; `run_nsga2_from_scratch` |
| 2 Evaluate | $f_1,f_2$ via Monte-Carlo over $K$ releases | `ExpectedModularityEvaluator.objectives` |
| 3 Non-dominated sorting | split into fronts $F_0,F_1,\dots$ | `fast_non_dominated_sort` |
| 4 Crowding distance | normalised neighbour gap per objective; boundaries = ∞ | `crowding_distance` |
| 5 Tournament selection | lower rank wins, ties → larger crowding distance | `binary_tournament_selection` |
| 6 SBX crossover | prob 0.9, $\eta_c=15$ | `sbx_crossover` |
| 7 Polynomial mutation | $\eta_m=20$, per-variable prob $\min(0.5,1/n_{var})$ | `polynomial_mutation` |
| 8 Environmental selection | merge parents+children, fill by fronts, split last front by crowding | `environmental_selection` |
| 9 Convergence tracking | per generation: hypervolume, front size, best $\mathbb E[Q]$, ε spread | `nsga2_optimizer._HistoryCallback` |
| 10 Pareto extraction | unique non-dominated points of the final population | `extract_pareto_front` |
| 11 Final comparison | NSGA-II vs dense grid, selected operating points | `compare_with_grid`, `select_solutions` |

The main run uses pymoo (`pymoo.algorithms.moo.nsga2.NSGA2`); a from-scratch NumPy implementation is run as a cross-check. Settings (all configurable): population 40, 30 generations, seed 42, SBX (0.9, 15), PM $\eta=20$, $K=20$ releases, bounds [0.1, 10].

## Stochastic evaluation and common random numbers
$\mathbb E[Q]$ is estimated from $K$ random releases. If fresh randomness is used for each evaluation, two nearly-equal $\varepsilon$ values can swap order by chance, which perturbs dominance and the observed front (selection also biases the *observed* $\mathbb E[Q]$ upward). The default evaluator draws the uniform matrix once and re-uses it for every $\varepsilon$ (CRN), making $f_2$ a deterministic function of $\varepsilon$. `figX2_common_vs_fresh_random_numbers` shows the fresh-noise variant for comparison. Even with CRN, $K=20$ means $f_2$ is an estimate: the per-release SD of $Q$ is up to ≈0.04 (`nsga2_grid_search.csv`, column `sd_q`).

## What the results mean (and don't)
* **One decision variable.** The search space is a 1-D interval. A 500-point grid costs 500 evaluations versus 1,200 for NSGA-II and is an essentially exact reference. In our run NSGA-II reaches **99.1 %** of the grid's hypervolume (from-scratch: 98.9 %); the remaining gap is the finite 40-point staircase, not a better or worse optimum.
* **The Pareto set is almost the whole curve.** $\mathbb E[Q(\varepsilon)]$ rises monotonically in $\varepsilon$ up to saturation, so every $\varepsilon$ before saturation is non-dominated (272 of 500 grid points, up to ε≈8.08); beyond that the curve is flat (up to ~1e-3 noise-level dips from finite $K$) and larger ε is dominated. NSGA-II therefore "finds" a spread of points along a known curve; it does not discover a global optimum, and which point to use is a *decision*, not an optimisation outcome.
* **Not DP end-to-end.** Scoring candidates against the original graph and selecting an operating point is data-dependent; see `differential_privacy.md`. The "selected" points (`results/csv/nsga2_selected_solutions.csv`, e.g. smallest ε with E[Q] ≥ 80 % of Q₀) use $Q_0$ of the original graph.

NSGA-II would become genuinely useful with additional decision variables (e.g. mechanism parameters, per-community budgets, generative-model hyper-parameters) or objectives that are expensive and non-monotone.
