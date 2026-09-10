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

__all__ = ["DesignBox", "inverse_design", "random_search", "adam_ascent"]


@dataclasses.dataclass(frozen=True)
class DesignBox:
    """The fabrication box: N layers with t in [t_lo, t_hi] um and
    n(lam0) in [n_lo, n_hi]."""

    n_layers: int
    t_lo: float
    t_hi: float
    n_lo: float
    n_hi: float

    def __post_init__(self):
        if self.n_layers < 1 or self.t_lo <= 0 or self.t_hi <= self.t_lo \
                or self.n_hi <= self.n_lo:
            raise ValueError("degenerate design box")

    def clip(self, t, n):
        return (np.clip(t, self.t_lo, self.t_hi),
                np.clip(n, self.n_lo, self.n_hi))

    def sample(self, rng, R):
        t = rng.uniform(self.t_lo, self.t_hi, (R, self.n_layers))
        n = rng.uniform(self.n_lo, self.n_hi, (R, self.n_layers))
        return t, n


def _merit_of(lam, t, n0, S, w, const, n_inc, n_sub):
    nlay = np.asarray(n0)[:, None] * np.asarray(S)[None, :]
    T = tmm.transmittance(lam, t, nlay, n_inc, n_sub)
    return float(tmm.merit(T, w, const))


def adam_ascent(lam, t, n0, S, w, const, box, n_iter=40, lr=4e-3,
                n_inc=1.0, n_sub=1.0, beta1=0.9, beta2=0.999, eps=1e-8):
    """Projected-Adam ascent from one seed; returns (t, n0, J, traj)."""
    t = np.asarray(t, float).copy()
    n0 = np.asarray(n0, float).copy()
    m = np.zeros(2 * box.n_layers)
    v = np.zeros(2 * box.n_layers)
    best = (t.copy(), n0.copy(), -np.inf)
    traj = []
    for it in range(1, n_iter + 1):
        J, gd, gn = adjoint.merit_and_grad(lam, t, n0, S, w, const,
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
    J = _merit_of(lam, t, n0, S, w, const, n_inc, n_sub)
    if J > best[2]:
        best = (t, n0, J)
    return best[0], best[1], best[2], np.asarray(traj)


def inverse_design(lam, S, w, const, box, n_probe=200, n_seed=4,
                   n_iter=40, lr=4e-3, n_inc=1.0, n_sub=1.0, seed=0):
    """Probe-seeded adjoint engine. Query budget = n_probe
    + 2 n_seed n_iter. Returns (t*, n*, J*, best-so-far trajectory)."""
    rng = np.random.default_rng(seed)
    tp, npr = box.sample(rng, n_probe)
    Jp = np.array([_merit_of(lam, tp[r], npr[r], S, w, const,
                             n_inc, n_sub) for r in range(n_probe)])
    order = np.argsort(Jp)[::-1]
    traj = list(np.maximum.accumulate(Jp))
    best = (None, None, -np.inf)
    for sidx in order[:n_seed]:
        t, n0, J, jt = adam_ascent(lam, tp[sidx], npr[sidx], S, w, const,
                                   box, n_iter=n_iter, lr=lr,
                                   n_inc=n_inc, n_sub=n_sub)
        for j in jt:
            traj.append(max(traj[-1], float(j)))
        if J > best[2]:
            best = (t, n0, J)
    return best[0], best[1], best[2], np.asarray(traj)


def random_search(lam, S, w, const, box, budget, n_inc=1.0, n_sub=1.0,
                  seed=1):
    """Equal-budget random-search baseline."""
    rng = np.random.default_rng(seed)
    t, n0 = box.sample(rng, budget)
    J = np.array([_merit_of(lam, t[r], n0[r], S, w, const, n_inc, n_sub)
                  for r in range(budget)])
    b = int(np.argmax(J))
    return t[b], n0[b], float(J[b]), np.maximum.accumulate(J)
