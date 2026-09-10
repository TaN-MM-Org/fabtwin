"""Yield statistics: the quantities manufacturing is decided by.

CVaR convention (Rockafellar and Uryasev, J. Risk 2, 21 (2000), lower
tail): CVaR_alpha is the mean of the q = ceil(alpha K) smallest merit
samples -- exactly, by sorting, with no interpolation, so the tests
can anchor it on hand-computable sets. `tail_statistics` bundles the
mean, std, lower percentile and CVaR of a merit sample; `pass_fail`
evaluates the hard spec of the FabGAN-ID study (worst-case in-band
leakage below a ceiling AND mean pass-band transmission above a
floor); `evaluate_under_process` scores a design by fresh Monte-Carlo
draws from any process (the reference simulator or any twin).
"""
from __future__ import annotations

import numpy as np

from . import tmm

__all__ = ["cvar", "tail_statistics", "pass_fail", "yield_fraction",
           "evaluate_under_process"]


def cvar(values, alpha):
    """Lower-tail CVaR: mean of the ceil(alpha K) smallest values."""
    v = np.asarray(values, dtype=float).ravel()
    if v.size == 0:
        raise ValueError("need at least one sample")
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must lie in (0, 1]")
    q = max(1, int(np.ceil(alpha * v.size)))
    return float(np.sort(v)[:q].mean())


def tail_statistics(values, alpha=0.05):
    """dict(mean, std, P (100*alpha percentile), CVaR) of a merit
    sample."""
    v = np.asarray(values, dtype=float).ravel()
    return dict(mean=float(v.mean()), std=float(v.std()),
                P=float(np.percentile(v, 100.0 * alpha)),
                CVaR=float(cvar(v, alpha)), alpha=float(alpha))


def pass_fail(T, stop_mask, pass_mask, leak_max=0.09, pass_min=0.90):
    """Hard filter spec per fabricated draw: worst in-band leakage
    max_stop T <= leak_max AND mean pass-band T >= pass_min.
    T : (..., L). Returns boolean array over the leading shape."""
    T = np.asarray(T, dtype=float)
    stop = np.asarray(stop_mask, bool)
    pas = np.asarray(pass_mask, bool)
    if not stop.any() or not pas.any():
        raise ValueError("empty stop or pass mask")
    leak = T[..., stop].max(axis=-1)
    ptrans = T[..., pas].mean(axis=-1)
    return (leak <= leak_max) & (ptrans >= pass_min)


def yield_fraction(T, stop_mask, pass_mask, leak_max=0.09, pass_min=0.90):
    """Fraction of draws passing the hard spec."""
    ok = pass_fail(T, stop_mask, pass_mask, leak_max, pass_min)
    return float(np.mean(ok))


def evaluate_under_process(sampler, t_um, n0, lam_um, shape, weights,
                           const, K, n_inc=1.0, n_sub=1.0, alpha=0.05):
    """Score a design by K fresh draws from `sampler(t, n, K)` ->
    (t_tilde (K,N), n_tilde (K,N)) -- the reference process, a twin,
    or anything else with that signature. Returns tail_statistics of
    the merit plus the merit samples."""
    tt, nt = sampler(t_um, n0, K)
    S = np.asarray(shape, dtype=float)
    if S.ndim == 1:
        S = np.broadcast_to(S[None, :], (len(t_um), S.size))
    J = np.empty(K)
    for k in range(K):
        nlay = np.asarray(nt[k])[:, None] * S
        T = tmm.transmittance(lam_um, tt[k], nlay, n_inc, n_sub)
        J[k] = tmm.merit(T, weights, const)
    out = tail_statistics(J, alpha)
    out["samples"] = J
    return out
