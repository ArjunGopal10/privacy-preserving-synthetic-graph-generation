# Differential privacy in this project

## Definition
A randomized mechanism $\mathcal M$ is **$\varepsilon$-differentially private** if for all neighbouring inputs $x\sim x'$ and all output sets $S$:
$\Pr[\mathcal M(x)\in S]\le e^{\varepsilon}\Pr[\mathcal M(x')\in S]$.

## Edge-DP versus node-DP
| | Neighbouring graphs differ in… | Protects | Cost |
|---|---|---|---|
| **Edge-DP** (this project) | one node pair (an edge present/absent) | the existence of *one relationship* | lower noise |
| Node-DP | one node and all its incident edges | the participation of a *whole person* | much higher noise / sensitivity |

This project uses **edge-DP**: the Karate Club nodes are fixed public labels (34 members) and we protect individual friendships. It does **not** provide node-level privacy, and it does not hide that a node exists.

## Why randomized response is epsilon-edge-DP
Two edge-neighbours differ in exactly one bit. Bits are perturbed independently, so only that bit's output distribution differs:
$\Pr[o\mid x]/\Pr[o\mid x']\in\{\frac{1-p}{p},\frac{p}{1-p}\}$ and $\frac{1-p}{p}=e^{\varepsilon}$. Hence one release is $\varepsilon$-edge-DP. This is verified numerically in `tests/test_randomized_response.py::test_dp_ratio_equals_epsilon`.
Deletion-only perturbation is **not** DP: any released edge must have existed (likelihood ratio unbounded) — included only as a contrast baseline.

## What the guarantee does not cover
* **Repeated releases compose.** $k$ independent releases at $\varepsilon$ cost $k\varepsilon$ (basic composition; `results/csv/privacy_budget_table.csv`).
* **Selecting epsilon with the original graph.** The NSGA-II/grid search scores candidates by modularity of releases of the *original* graph and then picks an operating point. That is data-dependent; the end-to-end pipeline is **not** shown to be DP. A data-independent rule for choosing $\varepsilon$ (or a DP selection mechanism with its own budget) is future work.
* **Identity / auxiliary information.** DP does not anonymise node identities and does not bound inference from outside knowledge beyond its prior-relative guarantee.

## Formal guarantee versus empirical indicators
| Kind | Examples | Interpretation |
|---|---|---|
| Formal | $\varepsilon$, $p$, $e^\varepsilon$ bound, composition | worst-case, holds for every input |
| Empirical risk indicators | edge precision, posterior belief, recall | computed on one known graph; depend on prior/density |
| Structural utility | density, clustering, degree KS, path length | similarity to the original |
| Community preservation | modularity under $P_0$, NMI, ARI, conductance | recoverability of groups |

The posterior-belief column (e.g. 0.15 at $\varepsilon=0.1$, 0.96 at $\varepsilon=5$ for prior 78/561≈0.139) is a Bayesian *illustration* under that specific prior: $\Pr[\text{real}\mid\text{observed edge}]=\frac{(1-p)\pi}{(1-p)\pi+p(1-\pi)}$.
