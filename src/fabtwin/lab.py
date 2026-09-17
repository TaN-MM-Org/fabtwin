"""Plan the calibration runs that feed the twin.

Everything else in this package assumes the fab traces already exist:
`errors_from_traces` turns them into error vectors, `GaussianTwin`
fits them, `twin_fidelity_report` grades the result. This module
answers the two questions a process engineer faces before any of
that: WHICH recipes should the calibration runs deposit, and HOW MANY
runs does a target twin accuracy cost?

* `design_recipes`: spread the calibration recipes over the design
  box so no region of recipe space goes unprobed -- a deterministic
  greedy maximin design (each added recipe maximizes its distance to
  the recipes already chosen, in box-normalized coordinates; the
  classic space-filling criterion, see M. E. Johnson, L. M. Moore and
  D. Ylvisaker, "Minimax and maximin distance designs", J. Statist.
  Plann. Inference 26, 131 (1990)). The greedy rule is transparent
  and recomputable -- the tests re-derive every pick -- but it is a
  good-practice heuristic, not a proof of the globally best design.
* `runs_for_twin_mean`: for a Gaussian twin, the standard error of
  each estimated mean-error component after M runs is exactly
  sqrt(variance / M), so the run count for a target accuracy is a
  closed form, not a search -- computed from a pilot set of traces
  (or any variance estimate) and refused when the pilot is too small
  to say anything.

Units are the package's own: thicknesses in micrometres, indices at
the platform's reference wavelength; error vectors as
`errors_from_traces` defines them (relative thickness, absolute
index).
"""
from __future__ import annotations

import numpy as np

from .design import DesignBox

__all__ = ["design_recipes", "runs_for_twin_mean"]


def design_recipes(box: DesignBox, n_recipes, n_candidates=512,
                   seed=0):
    """Choose well-spread calibration recipes inside the design box.

    Draws a candidate pool from the box, then greedily builds the
    design: the first recipe is the candidate nearest the pool's
    centroid, and each further recipe is the candidate farthest (in
    smallest-distance-to-the-chosen-set terms) from everything chosen
    so far -- maximin, in box-normalized coordinates so a micron of
    thickness and a unit of index are compared fairly. Frozen
    parameters (lo == hi) carry no distance, exactly.

    Deterministic for a given seed. Returns (recipes_t, recipes_n),
    each (n_recipes, N) -- exactly the shape
    `DepositionProcess.trace_dataset` takes.
    """
    n_recipes = int(n_recipes)
    n_candidates = int(n_candidates)
    if n_recipes < 1:
        raise ValueError("n_recipes must be >= 1")
    if n_candidates < n_recipes:
        raise ValueError("n_candidates must be >= n_recipes")
    rng = np.random.default_rng(seed)
    t, n = box.sample(rng, n_candidates)
    pool = np.hstack([t, n])                      # (R, 2N)
    lo = np.concatenate([np.broadcast_to(box.t_lo, (box.n_layers,)),
                         np.broadcast_to(box.n_lo, (box.n_layers,))])
    hi = np.concatenate([np.broadcast_to(box.t_hi, (box.n_layers,)),
                         np.broadcast_to(box.n_hi, (box.n_layers,))])
    span = hi - lo
    live = span > 0.0                             # frozen axes drop out
    x = np.zeros_like(pool)
    x[:, live] = (pool[:, live] - lo[live]) / span[live]
    if not np.any(live):
        raise ValueError("the box is fully frozen; nothing to design")

    centroid = x.mean(axis=0)
    first = int(np.argmin(np.sum((x - centroid) ** 2, axis=1)))
    chosen = [first]
    mind = np.sqrt(np.sum((x - x[first]) ** 2, axis=1))
    for _ in range(1, n_recipes):
        mind[chosen] = -np.inf
        nxt = int(np.argmax(mind))
        chosen.append(nxt)
        d_new = np.sqrt(np.sum((x - x[nxt]) ** 2, axis=1))
        mind = np.minimum(mind, d_new)
    idx = np.array(chosen)
    return t[idx].copy(), n[idx].copy()


def runs_for_twin_mean(target_se, pilot_x):
    """How many calibration runs does a target mean accuracy cost?

    For independent runs, the standard error of each estimated
    mean-error component after M runs is exactly
    sqrt(variance / M) -- so the smallest M bringing every
    component's standard error to `target_se` or below is

        M = ceil(max_i variance_i / target_se^2),

    a closed form, evaluated on the per-component variances of a
    pilot trace set (from `errors_from_traces`; at least 8 pilot
    runs, or the variance estimate itself is too noisy to plan with
    and this refuses).

    Returns (M, predicted_se) with predicted_se the (2N,) per-
    component standard errors at the returned M. The prediction is
    exact for the mean; covariance entries converge at the same 1/M
    rate but with larger constants, so treat M as a floor for those.
    """
    t = float(target_se)
    if not (np.isfinite(t) and t > 0.0):
        raise ValueError("target_se must be finite and positive")
    x = np.asarray(pilot_x, dtype=float)
    if x.ndim != 2 or x.shape[0] < 8:
        raise ValueError("need a pilot trace set of at least 8 runs "
                         "(M0, 2N): fewer runs cannot estimate the "
                         "variances this plan stands on")
    if not np.all(np.isfinite(x)):
        raise ValueError("pilot traces contain non-finite values")
    var = np.var(x, axis=0, ddof=1)
    if np.all(var == 0.0):
        raise ValueError("pilot traces have zero variance everywhere; "
                         "nothing to average over")
    m = max(1, int(np.ceil(float(var.max()) / t ** 2)))
    return m, np.sqrt(var / m)
