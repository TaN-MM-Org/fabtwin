"""Merits that are not linear in T, measured at any angle (new in
0.7.0).

The original design path scores a stack by J = w . T + const at
normal incidence. Here a merit is any differentiable function of the
spectra R, T and A = 1 - R - T, measured under one or more conditions
(angle of incidence and polarization), and an `OpticalModel` bundles
the merit, the conditions and the fixed absorption of each layer. The
exact gradient reaches the design through `fabtwin.gradients`, so
`inverse_design`, `robustify`, `random_search` and
`evaluate_under_process` accept a model wherever they accept linear
weights.

Merits provided (each returns J and its derivatives with respect to
every spectral value; higher J is better):

* `LinearMerit(weights, const, quantity)` -- w . X + const for X =
  "T", "R" or "A" (the original merit when X = "T" at normal
  incidence);
* `TargetMerit(target, quantity, weights)` -- minus the weighted mean
  squared distance to a target spectrum;
* `SpecMarginMerit(stop_mask, pass_mask, leak_max, pass_min,
  sharpness)` -- a smooth version of the pass/fail specification of
  `fabtwin.risk.pass_fail`, built so that J > 0 guarantees the
  specification is met (a log-sum-exp is never below the maximum, and
  a smooth minimum is never above the minimum; the tests assert the
  implication on random spectra);
* `FunctionMerit(func)` -- any user function returning (J, dJ/dR,
  dJ/dT, dJ/dA).

Every merit's gradient is checked against finite differences in the
tests.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .gradients import layer_indices, stack_rta_and_grads

__all__ = ["LinearMerit", "TargetMerit", "SpecMarginMerit",
           "FunctionMerit", "OpticalModel", "model_spectra",
           "model_merit_and_grad"]

_Q = ("R", "T", "A")


def _check_quantity(q):
    if q not in _Q:
        raise ValueError("quantity must be 'R', 'T' or 'A'")
    return q


class LinearMerit:
    """J = sum over conditions and wavelengths of w * X + const."""

    def __init__(self, weights, const=0.0, quantity="T"):
        self.w = np.asarray(weights, dtype=float)
        self.const = float(const)
        self.quantity = _check_quantity(quantity)

    def __call__(self, R, T, A):
        X = dict(R=R, T=T, A=A)[self.quantity]
        w = np.broadcast_to(self.w, X.shape)
        J = float(np.sum(w * X) + self.const)
        g = {q: np.zeros_like(X) for q in _Q}
        g[self.quantity] = np.array(w, dtype=float)
        return J, g["R"], g["T"], g["A"]


class TargetMerit:
    """J = - sum w (X - target)^2 / sum w (0 for a perfect match)."""

    def __init__(self, target, quantity="T", weights=None):
        self.target = np.asarray(target, dtype=float)
        self.quantity = _check_quantity(quantity)
        self.w = None if weights is None else np.asarray(weights, float)

    def __call__(self, R, T, A):
        X = dict(R=R, T=T, A=A)[self.quantity]
        tgt = np.broadcast_to(self.target, X.shape)
        w = np.ones_like(X) if self.w is None else \
            np.broadcast_to(self.w, X.shape)
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be >= 0 and not all zero")
        d = X - tgt
        J = float(-np.sum(w * d * d) / w.sum())
        g = {q: np.zeros_like(X) for q in _Q}
        g[self.quantity] = -2.0 * w * d / w.sum()
        return J, g["R"], g["T"], g["A"]


class SpecMarginMerit:
    """Smooth pass/fail margin: J > 0 guarantees max_stop T <= leak_max
    and mean_pass T >= pass_min in every condition.

    With s = `sharpness`, per condition c
        leak_c = (1/s) log sum_stop exp(s T)    (>= max_stop T)
        m1_c = leak_max - leak_c,  m2_c = mean_pass T - pass_min
    and J = -(1/s) log sum_c [exp(-s m1_c) + exp(-s m2_c)], which is at
    most every m1_c and m2_c. Larger s tracks the true margins more
    closely (the gap is at most log(n_stop)/s + log(2 C)/s).
    """

    def __init__(self, stop_mask, pass_mask, leak_max=0.09, pass_min=0.90,
                 sharpness=200.0):
        self.stop = np.asarray(stop_mask, bool)
        self.pas = np.asarray(pass_mask, bool)
        if not self.stop.any() or not self.pas.any():
            raise ValueError("empty stop or pass mask")
        self.leak_max = float(leak_max)
        self.pass_min = float(pass_min)
        self.s = float(sharpness)
        if not self.s > 0:
            raise ValueError("sharpness must be positive")

    def margins(self, T):
        """The exact (non-smooth) margins per condition, for reporting."""
        T = np.atleast_2d(T)
        return (self.leak_max - T[:, self.stop].max(axis=1),
                T[:, self.pas].mean(axis=1) - self.pass_min)

    def __call__(self, R, T, A):
        T2 = np.atleast_2d(T)
        if T2.shape[1] != self.stop.size or self.pas.size != self.stop.size:
            raise ValueError("stop and pass masks must have one entry per "
                             "wavelength")
        s = self.s
        Ts = T2[:, self.stop]
        mx = Ts.max(axis=1, keepdims=True)
        e = np.exp(s * (Ts - mx))
        leak = mx[:, 0] + np.log(e.sum(axis=1)) / s
        p_stop = e / e.sum(axis=1, keepdims=True)        # d leak / d T
        m1 = self.leak_max - leak
        m2 = T2[:, self.pas].mean(axis=1) - self.pass_min
        m = np.concatenate([m1, m2])
        mn = m.min()
        ex = np.exp(-s * (m - mn))
        J = float(mn - np.log(ex.sum()) / s)
        wgt = ex / ex.sum()                               # dJ/dm
        C = T2.shape[0]
        gT = np.zeros_like(T2)
        gT[:, self.stop] -= wgt[:C, None] * p_stop
        gT[:, self.pas] += wgt[C:, None] / self.pas.sum()
        gT = gT.reshape(np.shape(T))
        z = np.zeros_like(gT)
        return J, z, gT, z.copy()


class FunctionMerit:
    """Wrap a user function func(R, T, A) -> (J, dJ/dR, dJ/dT, dJ/dA);
    the arrays have the shape of the spectra, (conditions, L)."""

    def __init__(self, func):
        if not callable(func):
            raise ValueError("func must be callable")
        self.func = func

    def __call__(self, R, T, A):
        J, gR, gT, gA = self.func(R, T, A)
        return (float(J), np.asarray(gR, float), np.asarray(gT, float),
                np.asarray(gA, float))


@dataclasses.dataclass(frozen=True, eq=False)
class OpticalModel:
    """What is measured and how it is scored.

    merit : one of the merits above (called with spectra of shape
        (C, L), C = number of conditions).
    conditions : sequence of (theta0_rad, pol), pol "s", "p" or "u";
        default normal incidence.
    kext : None, (L,) or (N, L) extinction coefficients of the layers
        (fixed; the design varies thickness and the real index scale).
    """

    merit: object
    conditions: tuple = ((0.0, "s"),)
    kext: object = None

    def __post_init__(self):
        c = self.conditions
        if isinstance(c, tuple) and len(c) == 2 and isinstance(c[1], str):
            c = (c,)                  # a single (angle, pol) pair
        try:
            conds = tuple((float(a), str(p)) for a, p in c)
        except (TypeError, ValueError):
            raise ValueError("conditions must be a sequence of "
                             "(angle_rad, pol) pairs") from None
        if not conds:
            raise ValueError("need at least one condition")
        for a, p in conds:
            if not (0.0 <= a < np.pi / 2) or p not in ("s", "p", "u"):
                raise ValueError("each condition is (angle in [0, pi/2), "
                                 "'s' | 'p' | 'u')")
        object.__setattr__(self, "conditions", conds)
        if not callable(self.merit):
            raise ValueError("merit must be one of the fabtwin merits "
                             "(or callable like them)")


def model_spectra(model, lam_um, t_um, n0, shape, n_inc=1.0, n_sub=1.0,
                  grads=False):
    """Spectra (C, L) of R, T, A under every condition (and, with
    grads=True, the per-condition gradient dicts)."""
    n = layer_indices(n0, shape, model.kext)
    out = [stack_rta_and_grads(lam_um, t_um, n, n_inc, n_sub, a, p)
           for a, p in model.conditions]
    R = np.array([o["R"] for o in out])
    T = np.array([o["T"] for o in out])
    A = np.array([o["A"] for o in out])
    return (R, T, A, out) if grads else (R, T, A)


def model_merit_and_grad(model, lam_um, t_um, n0, shape, n_inc=1.0,
                         n_sub=1.0):
    """J and its exact gradients (dJ/dt, dJ/dn0) for an OpticalModel."""
    S = np.asarray(shape, dtype=float)
    t = np.asarray(t_um, dtype=float)
    if S.ndim == 1:
        S = np.broadcast_to(S[None, :], (t.size, S.size))
    R, T, A, out = model_spectra(model, lam_um, t_um, n0, shape, n_inc,
                                 n_sub, grads=True)
    J, gR, gT, gA = model.merit(R, T, A)
    gR, gT, gA = (np.broadcast_to(g, R.shape) for g in (gR, gT, gA))
    dt = np.zeros(t.size)
    dn = np.zeros(t.size)
    for c, o in enumerate(out):
        dt += o["dR_dt"] @ gR[c] + o["dT_dt"] @ gT[c] + o["dA_dt"] @ gA[c]
        dn += (o["dR_dn"] * S) @ gR[c] + (o["dT_dn"] * S) @ gT[c] \
            + (o["dA_dn"] * S) @ gA[c]
    return float(J), dt, dn
