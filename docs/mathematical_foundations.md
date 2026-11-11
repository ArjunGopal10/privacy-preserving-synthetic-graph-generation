# Mathematical foundations

## Graph and pair representation
Let $G=(V,E)$ be an undirected simple graph with $n=|V|$ and $m=|E|$. There are
$N=\binom{n}{2}=n(n-1)/2$ unordered node pairs (561 for $n=34$). $G$ is identified with the bit vector
$x\in\{0,1\}^N$, $x_{ij}=1$ iff $\{i,j\}\in E$ (upper triangle of the adjacency matrix, `numpy.triu_indices(n,1)` order).

## Randomized response
Each bit is flipped independently with probability
$$p(\varepsilon)=\frac{1}{1+e^{\varepsilon}}\in(0,\tfrac12].$$
$\varepsilon\to0$: $p\to\tfrac12$ (pure noise); $\varepsilon\to\infty$: $p\to0$ (exact copy).
Expected release: kept real edges $m(1-p)$, deleted $mp$, added fake edges $(N-m)p$, total $m(1-p)+(N-m)p$.

## Modularity under a fixed partition
For a partition $P=\{C_1,\dots,C_k\}$ and graph $H$ with $m_H$ edges,
$$Q(H;P)=\sum_{c}\left[\frac{L_c}{m_H}-\left(\frac{d_c}{2m_H}\right)^2\right],\qquad Q:=0\text{ if }m_H=0,$$
where $L_c$ = edges inside $C_c$ and $d_c$ = sum of degrees in $C_c$. $Q\in[-\tfrac12,1)$. Our implementation is tested against NetworkX.

## Optimisation problem
Decision variable $\varepsilon\in[\varepsilon_{\min},\varepsilon_{\max}]=[0.1,10]$.
$$f_1(\varepsilon)=\varepsilon,\qquad f_2(\varepsilon)=-\mathbb{E}[Q(H_\varepsilon;P_0)]\approx-\frac1K\sum_{k=1}^K Q(H^{(k)}_\varepsilon;P_0).$$
$P_0$ = Louvain partition of the *original* graph, computed once. Both objectives are minimised. See `nsga2_explanation.md`.

## Pareto dominance
$a$ dominates $b$ ($a\prec b$) iff $f_i(a)\le f_i(b)$ for all $i$ and $f_j(a)<f_j(b)$ for some $j$. The Pareto set is the set of non-dominated points; its image is the Pareto front.
Hypervolume (2-D): area of objective space dominated by the front and bounded by a reference point (here $(10,\,0.1)$).

## Empirical edge indicators (not DP metrics)
With $E_G$ original and $E_H$ released edge sets:
$\text{precision}=|E_G\cap E_H|/|E_H|$, $\text{recall}=|E_G\cap E_H|/|E_G|$, $F_1=2|E_G\cap E_H|/(|E_G|+|E_H|)$, $J=|E_G\cap E_H|/|E_G\cup E_H|$.
Under RR their plug-in values are precision $\approx m(1-p)/[m(1-p)+(N-m)p]$, recall $=1-p$, $J\approx m(1-p)/[m+(N-m)p]$ (verified empirically in `results/csv/privacy_summary.csv`).
