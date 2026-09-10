"""Pathwise CVaR robustification through twin and physics at once.

The design a manufacturer needs is not argmax J(theta) but the design
whose worst alpha-fraction of fabricated outcomes is best (Mahim et
al., IEEE Sensors J., 2026, Eq. 1). Because a reparameterized twin is
a deterministic map x(z) of Gaussian latents, the empirical CVaR of
the fabricated merit is a piecewise-smooth function of the design,
and its gradient flows through BOTH the corruption and the exact
transfer-matrix adjoint -- pathwise, with no surrogate. This module
implements that ascent in pure NumPy for any affinely
reparameterized twin (the Gaussian twins of `fabtwin.twins`); the
learned WGAN-GP twin ascends the identical objective through JAX in
`fabtwin.twin_jax`.

For one latent draw z_k: x_k = mu + L z_k, the fabricated design is
t_k = clip(t (1 + x_k[:N])), n_k = clip(n0 + x_k[N:]), and

    dJ_k/dt   = dJ/dt_k . (1 + x_k[:N]) . [inside clip box]
    dJ_k/dn0  = dJ/dn_k . [inside clip box]

exactly; the empirical CVaR gradient averages these over the tail
set, and the mean - beta sigma alternative weighs all draws. With
the latent batch FROZEN both objectives are deterministic, so the
tests anchor the whole chain against central finite differences --
no stochastic hand-waving.

Following the paper, the tail statistic is estimated from K draws
per step with a FRESH batch each step (re-sampling the tail set
rather than freezing it); the finite-K tail selection makes the
estimator biased for the population gradient with bias vanishing as
K grows, which the paper characterizes empirically rather than
claiming unbiasedness -- so does this docstring.
"""
from __future__ import annotations

import numpy as np

from . import adjoint, tmm

__all__ = ["cvar_objective_and_grad", "robustify"]


def _fab_clip_bounds(box):
    # fabricated devices may exceed the design box; the paper clips to
    # a wider physical window
    return 0.5 * box.t_lo, 2.0 * box.t_hi, box.n_lo - 0.20, box.n_hi + 0.20


def cvar_objective_and_grad(lam, t, n0, S, w, const, twin, z, alpha, box,
                            n_inc=1.0, n_sub=1.0, mean_variance=False,
                            beta=1.0):
    """Empirical CVaR_alpha (or mean - beta*std) of the fabricated
    merit over the latent batch z (K, 2N), with its exact gradient
    with respect to (t, n0). Deterministic for fixed z."""
    t = np.asarray(t, float)
    n0 = np.asarray(n0, float)
    N = t.size
    x = twin.transform(z)                     # (K, 2N)
    K = x.shape[0]
    t_lo, t_hi, n_lo, n_hi = _fab_clip_bounds(box)

    J = np.empty(K)
    gT = np.empty((K, N))
    gN = np.empty((K, N))
    for k in range(K):
        tk_raw = t * (1.0 + x[k, :N])
        nk_raw = n0 + x[k, N:]
        tk = np.clip(tk_raw, t_lo, t_hi)
        nk = np.clip(nk_raw, n_lo, n_hi)
        Jk, gdk, gnk = adjoint.merit_and_grad(lam, tk, nk, S, w, const,
                                              n_inc=n_inc, n_sub=n_sub)
        J[k] = Jk
        in_t = (tk_raw > t_lo) & (tk_raw < t_hi)
        in_n = (nk_raw > n_lo) & (nk_raw < n_hi)
        gT[k] = gdk * (1.0 + x[k, :N]) * in_t
        gN[k] = gnk * in_n
    if mean_variance:
        Jbar = J.mean()
        sd = J.std()
        # d/dtheta [mean - beta*std]; std treated as sqrt of the
        # biased variance, its exact derivative
        wgt = np.full(K, 1.0 / K)
        if sd > 0:
            wgt = wgt - beta * (J - Jbar) / (K * sd)
        val = Jbar - beta * sd
    else:
        q = max(1, int(np.ceil(alpha * K)))
        tail = np.argsort(J)[:q]
        wgt = np.zeros(K)
        wgt[tail] = 1.0 / q
        val = float(J[tail].mean())
    return val, wgt @ gT, wgt @ gN, J


def robustify(lam, t0, n00, S, w, const, twin, box, alpha=0.05, K=128,
              steps=150, lr=2e-3, seed=0, n_inc=1.0, n_sub=1.0,
              mean_variance=False, beta=1.0):
    """Projected stochastic CVaR ascent from a nominal design.

    A fresh latent minibatch z ~ N(0, I) is drawn each step (the
    paper's estimator); Adam with projection to the design box.
    Returns (t_rob, n_rob, history of (step, objective))."""
    rng = np.random.default_rng(seed)
    t = np.asarray(t0, float).copy()
    n0 = np.asarray(n00, float).copy()
    Np = box.n_layers
    m = np.zeros(2 * Np)
    v = np.zeros(2 * Np)
    hist = []
    for it in range(1, steps + 1):
        z = rng.normal(size=(K, twin.dim))
        val, gt, gn, _ = cvar_objective_and_grad(
            lam, t, n0, S, w, const, twin, z, alpha, box,
            n_inc=n_inc, n_sub=n_sub, mean_variance=mean_variance,
            beta=beta)
        g = np.concatenate([gt, gn])
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        mh = m / (1 - 0.9 ** it)
        vh = v / (1 - 0.999 ** it)
        t = t + lr * mh[:Np] / (np.sqrt(vh[:Np]) + 1e-8)
        n0 = n0 + lr * mh[Np:] / (np.sqrt(vh[Np:]) + 1e-8)
        t, n0 = box.clip(t, n0)
        if it % 10 == 0 or it == 1:
            hist.append((it, float(val)))
    return t, n0, hist
