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

An estimator without that bias (new in 0.7.0, `estimator="ru"`). The
lower-tail CVaR has the variational form of Rockafellar and Uryasev
(J. Risk 2, 21 (2000)),

    CVaR_alpha(J) = max over tau of  tau - E[(tau - J)_+] / alpha,

maximized at tau = the alpha-quantile of J. The objective inside is
an EXPECTATION, so for fixed (design, tau) its minibatch gradient

    d/d design = E[1{J < tau} dJ/d design] / alpha,
    d/d tau    = 1 - P(J < tau) / alpha

is unbiased at every K for this objective at the current tau (and
the objective equals the CVaR when tau is the alpha-quantile, which
the joint ascent tracks); ascending design and tau together is
ordinary stochastic-gradient ascent on one objective.
The tests compare the average of many small-K gradients with a
large-sample reference for both estimators: the sorting estimator
shows its bias, this one does not. What it does not change: the
answer is still a local optimum, and still only as good as the twin.

Twin uncertainty (new in 0.7.0). A twin fitted to M traces is itself
uncertain. `TwinEnsemble` refits the Gaussian twin on bootstrap
resamples of the traces (B. Efron and R. J. Tibshirani, An
Introduction to the Bootstrap, Chapman & Hall (1993)) and spreads the
K draws of each step over the members, so a design is robustified
against the machine AND against what M runs could not pin down about
it.

Merits other than w . T + const, other angles, and absorbing layers
enter through `model=` (a `fabtwin.OpticalModel`).
"""
from __future__ import annotations

import numpy as np

from . import adjoint, tmm
from .merits import model_merit_and_grad
from .twins import GaussianTwin

__all__ = ["cvar_objective_and_grad", "robustify", "TwinEnsemble",
           "ru_objective_and_grad"]


class TwinEnsemble:
    """Bootstrap ensemble of Gaussian twins: member b is a
    `GaussianTwin` fitted to a resample (with replacement) of the M
    trace error vectors. Row k of a latent batch is transformed by
    member k mod B, so a batch of K >= B draws covers every member.
    Exposes the same `dim`, `transform`, `sample_errors` and `sample`
    as a `GaussianTwin`."""

    def __init__(self, x_traces, n_members=20, diagonal=False, seed=0,
                 jitter=1e-9):
        x = np.asarray(x_traces, float)
        if x.ndim != 2 or x.shape[0] < 4:
            raise ValueError("x_traces must be (M >= 4, 2N)")
        B = int(n_members)
        if B < 2:
            raise ValueError("n_members must be >= 2")
        rng = np.random.default_rng(seed)
        M = x.shape[0]
        self.members = [GaussianTwin(x[rng.integers(0, M, M)],
                                     diagonal=diagonal, jitter=jitter)
                        for _ in range(B)]
        self.dim = x.shape[1]

    def transform(self, z):
        z = np.asarray(z, float)
        one = z.ndim == 1
        z = np.atleast_2d(z)
        out = np.empty_like(z)
        B = len(self.members)
        for b, m in enumerate(self.members):
            out[b::B] = m.transform(z[b::B])
        return out[0] if one else out

    def conditional(self, observed, values):
        """Every member conditioned on the measured error components
        (`GaussianTwin.conditional`); returns a TwinEnsemble."""
        out = TwinEnsemble.__new__(TwinEnsemble)
        out.members = [m.conditional(observed, values)
                       for m in self.members]
        out.dim = self.dim
        return out

    def sample_errors(self, rng, K):
        return self.transform(rng.normal(size=(K, self.dim)))

    def sample(self, rng, t_um, n0, K):
        from .twins import apply_errors
        return apply_errors(t_um, n0, self.sample_errors(rng, K))


def _fab_clip_bounds(box):
    # fabricated devices may exceed the design box; the paper clips to
    # a wider physical window
    return 0.5 * box.t_lo, 2.0 * box.t_hi, box.n_lo - 0.20, box.n_hi + 0.20


def _draws(lam, t, n0, S, w, const, twin, z, box, n_inc, n_sub, model):
    """Merit J_k and pathwise gradients of every draw of the batch."""
    t = np.asarray(t, float)
    n0 = np.asarray(n0, float)
    N = t.size
    x = twin.transform(z)
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
        if model is None:
            Jk, gdk, gnk = adjoint.merit_and_grad(
                lam, tk, nk, S, w, const, n_inc=n_inc, n_sub=n_sub)
        else:
            Jk, gdk, gnk = model_merit_and_grad(
                model, lam, tk, nk, S, n_inc=n_inc, n_sub=n_sub)
        J[k] = Jk
        in_t = (tk_raw > t_lo) & (tk_raw < t_hi)
        in_n = (nk_raw > n_lo) & (nk_raw < n_hi)
        gT[k] = gdk * (1.0 + x[k, :N]) * in_t
        gN[k] = gnk * in_n
    return J, gT, gN


def ru_objective_and_grad(lam, t, n0, tau, S, w, const, twin, z, alpha,
                          box, n_inc=1.0, n_sub=1.0, model=None):
    """The Rockafellar-Uryasev objective tau - mean((tau - J)_+)/alpha
    over the latent batch z, with its gradients with respect to
    (t, n0, tau) -- unbiased estimates of the population gradients for
    fixed (t, n0, tau). Returns (value, g_t, g_n, g_tau, J)."""
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must lie in (0, 1]")
    J, gT, gN = _draws(lam, t, n0, S, w, const, twin, z, box, n_inc,
                       n_sub, model)
    K = J.size
    below = (J < tau).astype(float)
    val = float(tau - np.sum(np.maximum(tau - J, 0.0)) / (alpha * K))
    wgt = below / (alpha * K)
    g_tau = 1.0 - below.mean() / alpha
    return val, wgt @ gT, wgt @ gN, float(g_tau), J


def cvar_objective_and_grad(lam, t, n0, S, w, const, twin, z, alpha, box,
                            n_inc=1.0, n_sub=1.0, mean_variance=False,
                            beta=1.0, model=None):
    """Empirical CVaR_alpha (or mean - beta*std) of the fabricated
    merit over the latent batch z (K, 2N), with its exact gradient
    with respect to (t, n0). Deterministic for fixed z. model= takes a
    `fabtwin.OpticalModel` (then w, const are ignored)."""
    from .design import _check_model
    _check_model(w, model)
    t = np.asarray(t, float)
    J, gT, gN = _draws(lam, t, n0, S, w, const, twin, z, box, n_inc,
                       n_sub, model)
    K = J.size
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
              mean_variance=False, beta=1.0, estimator="sort",
              model=None):
    """Projected stochastic CVaR ascent from a nominal design.

    A fresh latent minibatch z ~ N(0, I) is drawn each step; Adam with
    projection to the design box. estimator="sort" is the paper's
    estimator (mean of the ceil(alpha K) worst draws; biased at finite
    K); estimator="ru" ascends the Rockafellar-Uryasev objective with
    its unbiased minibatch gradient (module docstring), carrying the
    threshold tau along. twin may be a `GaussianTwin` or a
    `TwinEnsemble`; model= takes a `fabtwin.OpticalModel`.
    Returns (t_rob, n_rob, history of (step, objective))."""
    if estimator not in ("sort", "ru"):
        raise ValueError("estimator must be 'sort' or 'ru'")
    if estimator == "ru" and mean_variance:
        raise ValueError("estimator='ru' is for CVaR, not mean-variance")
    from .design import _check_model
    _check_model(w, model)
    rng = np.random.default_rng(seed)
    t = np.asarray(t0, float).copy()
    n0 = np.asarray(n00, float).copy()
    Np = box.n_layers
    ru = estimator == "ru"
    dimv = 2 * Np + (1 if ru else 0)
    m = np.zeros(dimv)
    v = np.zeros(dimv)
    hist = []
    tau = None
    for it in range(1, steps + 1):
        z = rng.normal(size=(K, twin.dim))
        if ru:
            if tau is None:        # start at the first batch's quantile
                J0, _, _ = _draws(lam, t, n0, S, w, const, twin, z, box,
                                  n_inc, n_sub, model)
                tau = float(np.quantile(J0, alpha))
            val, gt, gn, gtau, _ = ru_objective_and_grad(
                lam, t, n0, tau, S, w, const, twin, z, alpha, box,
                n_inc=n_inc, n_sub=n_sub, model=model)
            g = np.concatenate([gt, gn, [gtau]])
        else:
            val, gt, gn, _ = cvar_objective_and_grad(
                lam, t, n0, S, w, const, twin, z, alpha, box,
                n_inc=n_inc, n_sub=n_sub, mean_variance=mean_variance,
                beta=beta, model=model)
            g = np.concatenate([gt, gn])
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        mh = m / (1 - 0.9 ** it)
        vh = v / (1 - 0.999 ** it)
        step = lr * mh / (np.sqrt(vh) + 1e-8)
        t = t + step[:Np]
        n0 = n0 + step[Np:2 * Np]
        if ru:
            tau = tau + step[2 * Np]
        t, n0 = box.clip(t, n0)
        if it % 10 == 0 or it == 1:
            hist.append((it, float(val)))
    return t, n0, hist
