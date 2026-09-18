"""Distribution-free prediction intervals for twin predictions (new
in v0.6).

A twin's error bars are only as good as the twin. Split conformal
prediction makes a weaker promise that needs NO trust in the twin at
all: score how wrong the prediction was on held-out runs the twin
never trained on (for instance |measured - predicted| of a merit or
a thickness), take

    q = the ceil((n + 1)(1 - alpha))-th smallest of the n scores,

and a new run, exchangeable with the held-out ones, lands within q
of its prediction with probability at least 1 - alpha -- exactly, at
finite n, whatever the process and whatever the twin (V. Vovk, A.
Gammerman and G. Shafer, Algorithmic Learning in a Random World,
Springer (2005); J. Lei et al., J. Am. Stat. Assoc. 113, 1094
(2018); A. N. Angelopoulos and S. Bates, arXiv:2107.07511). For
continuous scores the coverage is also at most 1 - alpha + 1/(n+1),
with the exact value ceil((n+1)(1-alpha))/(n+1) -- a rank statement
the tests verify by seeded simulation against the closed form.

This is the honest companion to `twin_fidelity_report`: the report
grades how well the twin matches held-out data in distribution; the
conformal quantile converts the same held-out data into a guarantee
a production run can lean on.

Honest limits, stated plainly: the guarantee is marginal (on
average, not conditional on one particular recipe), and it needs
exchangeability -- held-out runs from before a tool drift do not
certify runs after it, which is the same retraining caveat the twin
itself carries.
"""
from __future__ import annotations

import numpy as np

__all__ = ["conformal_quantile", "conformal_interval",
           "conformal_coverage_exact"]


def conformal_quantile(scores, alpha=0.1):
    """The split-conformal quantile of held-out error scores.

    scores : (n,) nonnegative held-out error scores (e.g.
        |measured - predicted|).
    alpha : miscoverage level (0.1 = 90% intervals).

    Refuses when n is too small for the level: the required rank
    ceil((n+1)(1-alpha)) must exist among the n scores.
    """
    s = np.asarray(scores, dtype=float).ravel()
    if s.size < 1 or not np.all(np.isfinite(s)) or np.any(s < 0.0):
        raise ValueError("scores must be finite and >= 0 (use "
                         "absolute errors)")
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    n = s.size
    k = int(np.ceil((n + 1) * (1.0 - a)))
    if k > n:
        need = int(np.ceil((1.0 - a) / a))
        raise ValueError(
            f"{n} held-out scores cannot certify level {1 - a:.3g}: "
            f"the required rank {k} exceeds n. Hold out at least "
            f"{need} runs, or lower the confidence")
    return float(np.sort(s)[k - 1])


def conformal_interval(prediction, q):
    """The interval the guarantee applies to: prediction +/- q."""
    pred = np.asarray(prediction, dtype=float)
    qv = float(q)
    if not (np.isfinite(qv) and qv >= 0.0):
        raise ValueError("q must be finite and >= 0")
    return pred - qv, pred + qv


def conformal_coverage_exact(n, alpha):
    """The exact marginal coverage for continuous scores:
    ceil((n+1)(1-alpha)) / (n+1), always inside
    [1-alpha, 1-alpha + 1/(n+1)]."""
    n = int(n)
    a = float(alpha)
    if n < 1 or not (0.0 < a < 1.0):
        raise ValueError("need n >= 1 and alpha in (0, 1)")
    k = int(np.ceil((n + 1) * (1.0 - a)))
    if k > n:
        raise ValueError("level not certifiable at this n (see "
                         "conformal_quantile)")
    return k / (n + 1.0)
