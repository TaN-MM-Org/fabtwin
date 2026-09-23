"""Probe-seeded adjoint inverse design -- the nominal engine.

Global random probing identifies promising basins; projected-Adam
ascent driven by the exact hand adjoint exploits them (Mahim et al.,
IEEE Sensors J., 2026, Sec. VI-A: on the paper's benchmark this
engine reaches the best known merit in 360 solver queries where
random search does not match it within 30,000). Query accounting
follows the paper: one probe = one forward solve; one gradient
iteration = one forward + one adjoint solve.

Anchors asserted in the tests rather than stated: on the single-layer
antireflection problem, whose optimum is the closed-form quarter-wave
coating n = sqrt(n_inc n_sub), t = lam0 / (4 n), the engine recovers
the analytic optimum and the adjoint gradient vanishes there.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from . import adjoint, tmm
from .merits import model_merit_and_grad

__all__ = ["DesignBox", "inverse_design", "random_search", "adam_ascent"]


@dataclasses.dataclass(frozen=True, eq=False)
class DesignBox:
    """The fabrication box: N layers with t in [t_lo, t_hi] um and
    n(lam0) in [n_lo, n_hi].

    Each bound is a scalar (one window for every layer, the original
    behavior) or an (N,) array (per-layer windows, new in v0.3). A
    bound pair with lo == hi FREEZES that parameter: the optimizers
    project onto the box every step, so a frozen parameter never
    moves -- which is how a real multi-material stack, whose indices
    are deposited materials rather than design variables, enters the
    same loop. `DesignBox.thickness_only` builds that common case.
    Inverted bounds, non-positive thicknesses and a fully frozen box
    (nothing left to design) are refused.
    """

    n_layers: int
    t_lo: float
    t_hi: float
    n_lo: float
    n_hi: float

    def __post_init__(self):
        if int(self.n_layers) < 1:
            raise ValueError("degenerate design box")
        vals = {}
        for name in ("t_lo", "t_hi", "n_lo", "n_hi"):
            v = np.asarray(getattr(self, name), dtype=float)
            if v.ndim not in (0, 1) or (v.ndim == 1
                                        and v.shape != (self.n_layers,)):
                raise ValueError(f"{name} must be a scalar or an "
                                 f"({self.n_layers},) array")
            if not np.all(np.isfinite(v)):
                raise ValueError(f"{name} must be finite")
            if v.ndim == 1:
                object.__setattr__(self, name, v.copy())
            else:
                object.__setattr__(self, name, float(v))
            vals[name] = v
        if np.any(vals["t_lo"] <= 0.0) \
                or np.any(vals["t_hi"] < vals["t_lo"]) \
                or np.any(vals["n_hi"] < vals["n_lo"]):
            raise ValueError("degenerate design box")
        if np.all(vals["t_hi"] == vals["t_lo"]) \
                and np.all(vals["n_hi"] == vals["n_lo"]):
            raise ValueError("fully frozen design box: nothing left "
                             "to design")

    @classmethod
    def thickness_only(cls, n_layers, t_lo, t_hi, n_fixed):
        """The multi-material case: thicknesses free in [t_lo, t_hi],
        indices frozen at n_fixed (scalar or (N,), e.g. the
        alternating nH/nL profile of a two-material mirror)."""
        n_fixed = np.asarray(n_fixed, dtype=float)
        return cls(n_layers, t_lo, t_hi, n_fixed, n_fixed)

    def clip(self, t, n):
        return (np.clip(t, self.t_lo, self.t_hi),
                np.clip(n, self.n_lo, self.n_hi))

    def sample(self, rng, R):
        t = rng.uniform(np.broadcast_to(self.t_lo, (self.n_layers,)),
                        np.broadcast_to(self.t_hi, (self.n_layers,)),
                        (R, self.n_layers))
        n = rng.uniform(np.broadcast_to(self.n_lo, (self.n_layers,)),
                        np.broadcast_to(self.n_hi, (self.n_layers,)),
                        (R, self.n_layers))
        return t, n


def _check_model(w, model):
    if model is None and w is None:
        raise ValueError("give linear weights (w, const) or an "
                         "OpticalModel via model=")
    if model is not None and w is not None:
        raise ValueError("give either linear weights (w, const) or "
                         "model=, not both (pass w=None, const=None "
                         "with a model)")


def _merit_of(lam, t, n0, S, w, const, n_inc, n_sub, model=None):
    if model is not None:
        from .merits import model_spectra
        R, T, A = model_spectra(model, lam, t, n0, S, n_inc, n_sub)
        return float(model.merit(R, T, A)[0])
    S = np.asarray(S)
    n0 = np.asarray(n0)
    nlay = n0[:, None] * (S[None, :] if S.ndim == 1 else S)
    T = tmm.transmittance(lam, t, nlay, n_inc, n_sub)
    return float(tmm.merit(T, w, const))


def adam_ascent(lam, t, n0, S, w, const, box, n_iter=40, lr=4e-3,
                n_inc=1.0, n_sub=1.0, beta1=0.9, beta2=0.999, eps=1e-8,
                model=None):
    """Projected-Adam ascent from one seed; returns (t, n0, J, traj).

    model : optional `fabtwin.OpticalModel` (any angle, absorbing
    layers, nonlinear merit); w and const are then ignored (pass
    None)."""
    _check_model(w, model)
    t = np.asarray(t, float).copy()
    n0 = np.asarray(n0, float).copy()
    m = np.zeros(2 * box.n_layers)
    v = np.zeros(2 * box.n_layers)
    best = (t.copy(), n0.copy(), -np.inf)
    traj = []
    for it in range(1, n_iter + 1):
        if model is None:
            J, gd, gn = adjoint.merit_and_grad(lam, t, n0, S, w, const,
                                               n_inc=n_inc, n_sub=n_sub)
        else:
            J, gd, gn = model_merit_and_grad(model, lam, t, n0, S,
                                             n_inc=n_inc, n_sub=n_sub)
        traj.append(J)
        if J > best[2]:
            best = (t.copy(), n0.copy(), J)
        g = np.concatenate([gd, gn])
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        mh = m / (1 - beta1 ** it)
        vh = v / (1 - beta2 ** it)
        step = lr * mh / (np.sqrt(vh) + eps)
        t = t + step[:box.n_layers]
        n0 = n0 + step[box.n_layers:]
        t, n0 = box.clip(t, n0)
    J = _merit_of(lam, t, n0, S, w, const, n_inc, n_sub, model)
    if J > best[2]:
        best = (t, n0, J)
    return best[0], best[1], best[2], np.asarray(traj)


def inverse_design(lam, S, w, const, box, n_probe=200, n_seed=4,
                   n_iter=40, lr=4e-3, n_inc=1.0, n_sub=1.0, seed=0,
                   model=None):
    """Probe-seeded adjoint engine. Query budget = n_probe
    + 2 n_seed n_iter. Returns (t*, n*, J*, best-so-far trajectory).
    With model= (a `fabtwin.OpticalModel`) the merit, angles and
    absorption come from the model and w, const are ignored."""
    _check_model(w, model)
    rng = np.random.default_rng(seed)
    tp, npr = box.sample(rng, n_probe)
    Jp = np.array([_merit_of(lam, tp[r], npr[r], S, w, const,
                             n_inc, n_sub, model) for r in range(n_probe)])
    order = np.argsort(Jp)[::-1]
    traj = list(np.maximum.accumulate(Jp))
    best = (None, None, -np.inf)
    for sidx in order[:n_seed]:
        t, n0, J, jt = adam_ascent(lam, tp[sidx], npr[sidx], S, w, const,
                                   box, n_iter=n_iter, lr=lr,
                                   n_inc=n_inc, n_sub=n_sub, model=model)
        for j in jt:
            traj.append(max(traj[-1], float(j)))
        if J > best[2]:
            best = (t, n0, J)
    return best[0], best[1], best[2], np.asarray(traj)


def random_search(lam, S, w, const, box, budget, n_inc=1.0, n_sub=1.0,
                  seed=1, model=None):
    """Equal-budget random-search baseline."""
    _check_model(w, model)
    rng = np.random.default_rng(seed)
    t, n0 = box.sample(rng, budget)
    J = np.array([_merit_of(lam, t[r], n0[r], S, w, const, n_inc, n_sub,
                            model) for r in range(budget)])
    b = int(np.argmax(J))
    return t[b], n0[b], float(J[b]), np.maximum.accumulate(J)
