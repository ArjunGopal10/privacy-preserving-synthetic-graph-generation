"""pymoo problem definition: one decision variable (epsilon), two objectives."""
from __future__ import annotations

import numpy as np
from pymoo.core.problem import ElementwiseProblem

from .fitness_functions import ExpectedModularityEvaluator


class EpsilonPrivacyUtilityProblem(ElementwiseProblem):
    """min [ epsilon, -E[Q(H; P0)] ]  subject to  eps_min <= epsilon <= eps_max.

    NOTE: because the evaluator looks at the original graph, selecting epsilon through this
    problem is a data-dependent step; it is NOT covered by the epsilon-DP guarantee of a single
    release (see docs/differential_privacy.md).
    """

    def __init__(self, evaluator: ExpectedModularityEvaluator, eps_min: float = 0.1,
                 eps_max: float = 10.0):
        if not 0 <= eps_min < eps_max:
            raise ValueError("Need 0 <= eps_min < eps_max.")
        super().__init__(n_var=1, n_obj=2, n_ieq_constr=0, xl=np.array([eps_min]),
                         xu=np.array([eps_max]))
        self.evaluator = evaluator

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.evaluator.objectives(float(x[0]))
