# Randomized response

Warner (1965) introduced randomized response for sensitive survey questions; applied to graphs it flips every adjacency bit independently with probability $p=1/(1+e^{\varepsilon})$.

## Algorithm
1. Build the upper-triangular bit vector $x\in\{0,1\}^{N}$ of the graph.
2. Draw $u_{ij}\sim U(0,1)$ for each pair; flip $x_{ij}$ iff $u_{ij}<p$.
3. Rebuild the graph on the same node set.

## Small example (n = 4, N = 6, epsilon = 1 → p ≈ 0.269)
Original edges: {01, 12, 23}. Expect about $6\times0.269\approx1.6$ flips: each of the 3 real edges survives w.p. 0.731 and each of the 3 non-edges becomes a fake edge w.p. 0.269.

## Why epsilon controls the trade-off
Small $\varepsilon$ → $p$ near ½ → released graph ≈ Erdős–Rényi noise (at $\varepsilon=0.1$ the release has ≈3.5× as many edges as the original in our run; community structure is gone). Large $\varepsilon$ → $p\to0$ → faithful copy but weak privacy ($\Pr[\text{real}\mid\text{edge}]\to1$).

## Dense mechanism, scalability
RR touches all $N=O(n^2)$ pairs, so it is only practical for small graphs (fine for 34 nodes; infeasible for ego-Facebook-scale graphs without a sparse/scalable edge-DP mechanism).

## Common random numbers (CRN)
All releases are `flip = (U < p)` with a stored $U$. Re-using $U$ across epsilons nests the flip sets (tested in `test_common_random_numbers_are_nested`), which removes sampling noise from epsilon-to-epsilon comparisons without changing any single release's distribution.

Code: `src/graph_generation/randomized_response.py`.
