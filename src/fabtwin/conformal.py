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
           "conformal_coverage_exact", "mondrian_quantiles",
           "AdaptiveConformal"]


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


# ----------------------------------------------------------------------
# Per-group guarantees and drifting processes (new in 0.7.0)


def mondrian_quantiles(scores, groups, alpha=0.1):
    """Group-conditional ("Mondrian") split conformal: one quantile per
    group, from that group's held-out scores alone.

    Within each group the guarantee of `conformal_quantile` holds for
    new runs of THAT group (Vovk, Gammerman and Shafer (2005), Mondrian
    conformal predictors) -- for example per recipe, when each recipe
    has its own held-out runs. A group with too few runs for the level
    is refused (the message names it). Conditioning on every possible
    recipe at once, without runs of each, is impossible for any
    method of this kind: R. F. Barber, E. J. Candes, A. Ramdas and
    R. J. Tibshirani, Information and Inference 10, 455 (2021).

    scores : (n,) held-out scores; groups : (n,) hashable labels.
    Returns dict label -> q.
    """
    s = np.asarray(scores, dtype=float).ravel()
    labels = list(groups)
    if len(labels) != s.size:
        raise ValueError("scores and groups must have the same length")
    idx = {}
    for i, lab in enumerate(labels):
        key = lab.item() if hasattr(lab, "item") else lab
        idx.setdefault(key, []).append(i)
    out = {}
    for key, ii in idx.items():
        try:
            out[key] = conformal_quantile(s[ii], alpha)
        except ValueError as exc:
            raise ValueError(f"group {key!r}: {exc}") from None
    return out


class AdaptiveConformal:
    """Adaptive conformal inference for a process that drifts (I. Gibbs
    and E. Candes, "Adaptive Conformal Inference Under Distribution
    Shift", NeurIPS 2021).

    At each new run the interval half-width is the (1 - alpha_t)
    quantile of the most recent `window` scores (infinite when
    alpha_t <= 0, empty when alpha_t >= 1). After the run's score s_t
    is seen, err_t = 1{s_t > half-width} and

        alpha_{t+1} = alpha_t + gamma (alpha - err_t).

    Whatever the sequence of scores -- drifting, jumping, adversarial
    -- alpha_t stays in [-gamma, 1 + gamma], and summing the updates
    gives, for every T,

        | (1/T) sum_t err_t - alpha | <= (max(alpha_1, 1 - alpha_1)
                                          + gamma) / (gamma T),

    a long-run guarantee with no exchangeability assumption (the tests
    check this deterministic bound on a drifting sequence). It is a
    frequency statement over time, not a probability for any single
    run.
    """

    def __init__(self, scores_init, alpha=0.1, gamma=0.005, window=200):
        s = np.asarray(scores_init, dtype=float).ravel()
        if s.size < 1 or not np.all(np.isfinite(s)) or np.any(s < 0):
            raise ValueError("initial scores must be finite and >= 0")
        if not (0.0 < alpha < 1.0) or not (gamma > 0.0):
            raise ValueError("need alpha in (0, 1) and gamma > 0")
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.window = int(window)
        if self.window < 1:
            raise ValueError("window must be >= 1")
        self.alpha_t = float(alpha)
        self.alpha_1 = float(alpha)
        self.scores = list(s[-self.window:])
        self.errors = []

    def half_width(self):
        """Current interval half-width q_t (inf or -inf at the ends)."""
        a = self.alpha_t
        if a <= 0.0:
            return np.inf
        if a >= 1.0:
            return -np.inf
        s = np.sort(self.scores)
        n = s.size
        k = int(np.ceil((n + 1) * (1.0 - a)))
        return float(s[k - 1]) if k <= n else np.inf

    def update(self, score):
        """Record one new run's score; returns whether it was missed."""
        sc = float(score)
        if not (np.isfinite(sc) and sc >= 0):
            raise ValueError("score must be finite and >= 0")
        err = float(sc > self.half_width())
        self.errors.append(err)
        self.alpha_t += self.gamma * (self.alpha - err)
        self.scores.append(sc)
        if len(self.scores) > self.window:
            self.scores.pop(0)
        return bool(err)

    def miss_rate(self):
        return float(np.mean(self.errors)) if self.errors else float("nan")

    def bound(self):
        """The guaranteed |miss rate - alpha| after the runs so far."""
        T = len(self.errors)
        if T == 0:
            return float("inf")
        return (max(self.alpha_1, 1 - self.alpha_1) + self.gamma) \
            / (self.gamma * T)
