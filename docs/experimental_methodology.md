# Experimental methodology

## Data
NetworkX Zachary Karate Club: 34 nodes, 78 edges, 561 node pairs (`data/sample/karate_club_edges.txt`; Zachary, 1977).

## Common protocol
* Seed 42 everywhere (`--seed`). Releases per epsilon $K=20$ (`--n-releases`); epsilon grid 0.1, 0.5, 1, 2, 3, 5, 8 (`--epsilons`).
* Releases for different epsilons share one uniform matrix (CRN, see `randomized_response.md`).
* Summaries are mean ± 95 % t-interval over the $K$ releases (error bars in figures); CSVs also hold SD.
* Every script writes `results/reports/*_metadata.json` (arguments, seed, Python/library versions, platform) and a figure manifest; every figure has its plotted data in `results/csv/figure_data/`.
* Scripts refuse to overwrite existing results unless `--overwrite` is passed.

## Experiments
| Script | Question | Outputs |
|---|---|---|
| `run_privacy_comparison.py` | how do formal and empirical privacy indicators change with ε? | Figs 1–8, `privacy_*.csv` |
| `run_community_comparison.py` | how much community structure survives (Protocol A and B, 5 detectors)? | Figs 9–17, `community_*.csv` |
| `run_nsga2.py` | what is the privacy–utility front, and does NSGA-II beat a grid? | Figs 18–23, `nsga2_*.csv` |
| `run_baseline_comparison.py` | how do RR, Laplace top-m and non-private baselines compare structurally? | Figs 24–30, `baseline_*.csv` |
| `run_complete_experiment.py` | all of the above + `experiment_report.md` | everything |

## Baselines (and DP status)
| Baseline | Formal edge-DP? | Assumption / caveat |
|---|---|---|
| Original | no | reference |
| Randomized response | **yes** (ε) | dense, one release |
| Laplace top-m (simplified TmF-style) | **yes** (ε split ½/½) | simplified re-implementation inspired by Nguyen et al. (2015), *not* an exact reproduction |
| Matched G(n,m) | no | uses the true edge count; ignores structure |
| Degree-preserving rewiring | no | leaks the exact degree sequence |
| Deletion-only | no | every released edge is real |

## Consistency check against the existing website
The site's table (computed by notebooks not available to us) and this repository's independent run agree within Monte-Carlo variation:

| ε | precision (site / here) | Q under P₀ (site / here) | NMI vs P₀ (site / here) |
|---|---|---|---|
| 0.1 | 0.15 / 0.151 | −0.014 / −0.013 | 0.08 / 0.086 |
| 2 | 0.54 / 0.550 | 0.160 / 0.175 | 0.30 / 0.325 |
| 3 | 0.77 / 0.762 | 0.266 / 0.268 | 0.45 / 0.503 |
| 5 | 0.96 / 0.953 | 0.368 / 0.360 | 0.78 / 0.745 |
| 8 | 1.00 / 0.999 | 0.385 / 0.384 | 1.00 / 0.988 |

Differences (largest 0.015 in Q at ε=2, 0.05 in NMI at ε=3) are of the order of sampling error for 20 releases and different random draws. The reference partition (17/13/4, $Q_0=0.3854$) is reproduced exactly.
The site's NSGA-II curves are labelled there as browser simulations; the pymoo results in `results/` are the real run and can replace them.

## Statistical caveats
Single small graph; $K=20$; CIs are over release noise only (not over graphs, seeds of P₀, or detection randomness). Girvan–Newman uses only the first 5 releases per ε. Detection seeds are fixed (= master seed), so detector randomness is not averaged out.
