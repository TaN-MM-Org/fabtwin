"""Correct a run while it is being deposited (new in 0.7.0).

When an in-situ monitor measures the layers deposited so far, the
layers still to come can be re-designed to make up for the errors
already made -- the classic re-optimization of the remaining layers
of a coating run. `reoptimize_remaining` does that with the package's
exact gradients: the deposited layers are frozen at their MEASURED
values and the remaining layers are re-designed, either for the best
nominal merit or, given a twin, for the best CVaR of what the machine
will still do to them. In the robust case the twin is first
conditioned on the measured errors (`GaussianTwin.conditional`), so
error components correlated with those already seen -- a drifting
rate, AR(1) thickness noise -- are predicted rather than treated as
fresh.

Which layers come first. Layer 0 of every fabtwin stack is the one
next to the incidence medium, layer N-1 the one on the substrate. A
coating is grown on the substrate, so by default (`first="substrate"`)
the deposited layers are the LAST n_done in array order; use
`first="incidence"` for a stack whose layer 0 is grown first (for
example a coating used with light arriving through the substrate, or
the simulated `DepositionProcess`, whose intermixing and correlated
noise run in array order).

This is a simple, well-defined correction rule, not the paper's
Stage 3 (specification-conditioned correction policies, which the
paper treats as exploratory and which remain outside this package).

What the tests hold it to: without a twin, the corrected recipe never
scores lower than the uncorrected one on the nominal merit given the
measured layers (the ascent starts from it and keeps the best point;
with a twin the result is the robust design, which may give up some
nominal merit for robustness); the deposited layers are returned
unchanged; and on the reference process the corrected runs end up
better on average than the same runs left uncorrected, with the
remaining layers' errors held identical.
"""
from __future__ import annotations

import numpy as np

from .design import DesignBox, _merit_of, adam_ascent

__all__ = ["reoptimize_remaining"]


def reoptimize_remaining(lam, t_recipe, n_recipe, S, w, const, n_done,
                         t_measured, n_measured, box, twin=None,
                         alpha=0.1, K=64, steps=60, n_iter=40, lr=4e-3,
                         n_inc=1.0, n_sub=1.0, model=None, seed=0,
                         estimator="ru", first="substrate"):
    """Re-design the layers not yet deposited.

    lam, S, w, const, n_inc, n_sub, model : as in `inverse_design`.
    t_recipe, n_recipe : (N,) the original recipe.
    n_done : how many layers are already deposited.
    t_measured, n_measured : (n_done,) their measured thickness and
        index, in array order (layer index increasing).
    first : "substrate" (default: the deposited layers are the last
        n_done in array order, next to the substrate) or "incidence"
        (the first n_done); see the module docstring.
    box : the `DesignBox` of the full stack (its bounds for the
        remaining layers are used).
    twin : optional `GaussianTwin` or `TwinEnsemble` of the machine;
        given, the remaining layers are robustified for CVaR_alpha of
        what the machine will still do to them (`robustify` with
        `estimator`, "ru" by default here), after the twin is
        conditioned on the measured errors.

    Returns dict(t, n) -- the full corrected recipe, deposited layers
    at their measured values -- and J_before, J_after: the nominal
    merit of the stack with the measured layers and the old, resp. new,
    remaining layers.
    """
    if first not in ("substrate", "incidence"):
        raise ValueError("first must be 'substrate' or 'incidence'")
    t_recipe = np.asarray(t_recipe, float)
    n_recipe = np.asarray(n_recipe, float)
    N = t_recipe.size
    m = int(n_done)
    if not (0 < m < N):
        raise ValueError("n_done must leave at least one layer on each "
                         "side (0 < n_done < N)")
    tm = np.asarray(t_measured, float)
    nm = np.asarray(n_measured, float)
    if tm.shape != (m,) or nm.shape != (m,) or np.any(tm <= 0):
        raise ValueError("t_measured and n_measured must be (n_done,), "
                         "thicknesses positive")
    if box.n_layers != N:
        raise ValueError("box and recipe disagree on the layer count")
    done = np.arange(m) if first == "incidence" else np.arange(N - m, N)

    def per(v):
        return np.broadcast_to(np.asarray(v, float), (N,)).copy()

    tlo, thi, nlo, nhi = (per(box.t_lo), per(box.t_hi), per(box.n_lo),
                          per(box.n_hi))
    t_start = t_recipe.copy()
    n_start = n_recipe.copy()
    t_start[done] = tm
    n_start[done] = nm
    # deposited layers frozen at the measured values
    tlo[done] = thi[done] = tm
    nlo[done] = nhi[done] = nm
    fbox = DesignBox(N, tlo, thi, nlo, nhi)
    t_start, n_start = fbox.clip(t_start, n_start)
    J0 = _merit_of(lam, t_start, n_start, S, w, const, n_inc, n_sub, model)
    t1, n1, J1, _ = adam_ascent(lam, t_start, n_start, S, w, const, fbox,
                                n_iter=n_iter, lr=lr, n_inc=n_inc,
                                n_sub=n_sub, model=model)
    if twin is not None:
        from .robust import robustify
        obs = np.concatenate([done, N + done])
        x_obs = np.concatenate([tm / t_recipe[done] - 1.0,
                                nm - n_recipe[done]])
        ctwin = twin.conditional(obs, x_obs)
        # in the robust objective the deposited layers stay at the
        # RECIPE, and the conditioned twin fixes their errors at the
        # measured ones, so recipe (1 + x) is the measured stack
        rlo, rhi, rnlo, rnhi = tlo.copy(), thi.copy(), nlo.copy(), \
            nhi.copy()
        rlo[done] = rhi[done] = t_recipe[done]
        rnlo[done] = rnhi[done] = n_recipe[done]
        rbox = DesignBox(N, rlo, rhi, rnlo, rnhi)
        ts, ns = t1.copy(), n1.copy()
        ts[done] = t_recipe[done]
        ns[done] = n_recipe[done]
        tr, nr, _ = robustify(lam, ts, ns, S, w, const, ctwin, rbox,
                              alpha=alpha, K=K, steps=steps, lr=lr / 2,
                              seed=seed, n_inc=n_inc, n_sub=n_sub,
                              model=model, estimator=estimator)
        t1, n1 = tr.copy(), nr.copy()
        t1[done] = tm
        n1[done] = nm
        J1 = _merit_of(lam, t1, n1, S, w, const, n_inc, n_sub, model)
    return dict(t=t1, n=n1, J_before=float(J0), J_after=float(J1))
