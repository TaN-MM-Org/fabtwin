"""Twin fidelity against held-out real traces -- scoring a process
model when there is no hidden simulator to draw fresh truth from.

The paper scores twins against a held-out ground-truth simulator
(Table I(A): |dP5|, |dCVaR5|, W1 of the induced merit distribution,
correlation error). A lab working from real in-situ monitor data has
no such oracle -- only its traces. The honest substitute, provided
here, is a held-out split: fit the twin on part of the traces, then
compare twin-generated error samples against the HELD-OUT real error
vectors, both marginally (pooled moments and correlations) and where
yield is decided (the lower tail of the merit distribution each
induces on a bank of designs, through the exact solver). These are
the same statistic types as the paper's Table I(A), computed between
two empirical samples rather than sample-vs-oracle.

Anchors asserted in the tests rather than stated: every distance is
exactly zero when a sample is compared against itself; the 1-D
Wasserstein distance agrees with SciPy's independent implementation
exactly; the tail statistics are the same `cvar`/percentile
conventions as `fabtwin.risk`; and a deliberately mean-shifted twin
scores strictly worse than a twin fitted to the same distribution.

Honest scope, stated plainly: a held-out split certifies that the
twin matches the data-generating process AS SAMPLED -- it cannot
certify behavior on error patterns the tool has never logged, and
the paper's yield-gain results are established within its virtual
setting. On real data this module tells you how faithful your twin
is to your fab's recorded behavior; it does not promise the fab will
keep behaving that way (the paper's Applicability paragraph: in
production the twin needs periodic retraining on fresh traces as the
tool drifts).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

from . import tmm
from .risk import cvar
from .twins import apply_errors

__all__ = ["moment_errors", "distribution_distances",
           "induced_merits", "twin_fidelity_report"]


def moment_errors(x_model, x_ref):
    """Pooled first/second-moment discrepancies between two error
    samples (K, 2N) and (M, 2N): max absolute mean difference,
    Frobenius error of the covariances, and Frobenius error of the
    correlation matrices (the paper's 'Corr. err')."""
    a = np.asarray(x_model, dtype=float)
    b = np.asarray(x_ref, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("samples must be (K, D) and (M, D) with one D")
    d_mean = float(np.abs(a.mean(0) - b.mean(0)).max())
    ca = np.cov(a.T)
    cb = np.cov(b.T)
    d_cov = float(np.linalg.norm(ca - cb))

    def corr(c):
        s = np.sqrt(np.clip(np.diag(c), 1e-300, None))
        return c / np.outer(s, s)

    d_corr = float(np.linalg.norm(corr(ca) - corr(cb)))
    return dict(d_mean=d_mean, d_cov=d_cov, d_corr=d_corr)


def distribution_distances(j_model, j_ref, alpha=0.05):
    """Tail-focused distances between two merit samples: |d mean|,
    |d P(100 alpha)|, |d CVaR_alpha| (the `fabtwin.risk` conventions)
    and the 1-D Wasserstein distance W1 (SciPy)."""
    a = np.asarray(j_model, dtype=float).ravel()
    b = np.asarray(j_ref, dtype=float).ravel()
    q = 100.0 * alpha
    return dict(
        d_mean=float(abs(a.mean() - b.mean())),
        d_P=float(abs(np.percentile(a, q) - np.percentile(b, q))),
        d_CVaR=float(abs(cvar(a, alpha) - cvar(b, alpha))),
        W1=float(wasserstein_distance(a, b)),
        alpha=float(alpha))


def induced_merits(x, t_um, n0, lam_um, shape, weights, const,
                   n_inc=1.0, n_sub=1.0, t_clip=None, n_clip=None):
    """Merit of one design under each error vector in x (K, 2N),
    through the exact solver. Optional (lo, hi) clips mirror the
    physical fabrication window used during robustification."""
    x = np.asarray(x, dtype=float)
    S = np.asarray(shape, dtype=float)
    if S.ndim == 1:
        S = np.broadcast_to(S[None, :], (len(t_um), S.size))
    tt, nn = apply_errors(t_um, n0, x)
    if t_clip is not None:
        tt = np.clip(tt, *t_clip)
    if n_clip is not None:
        nn = np.clip(nn, *n_clip)
    J = np.empty(x.shape[0])
    for k in range(x.shape[0]):
        T = tmm.transmittance(lam_um, tt[k], nn[k][:, None] * S,
                              n_inc, n_sub)
        J[k] = tmm.merit(T, weights, const)
    return J


def twin_fidelity_report(x_model, x_heldout, designs, lam_um, shape,
                         weights, const, n_inc=1.0, n_sub=1.0,
                         alpha=0.05, t_clip=None, n_clip=None):
    """The held-out fidelity report: pooled moment errors plus the
    induced-merit distances of `distribution_distances`, averaged over
    a bank of designs.

    x_model : (K, 2N) error vectors sampled from the twin under test.
    x_heldout : (M, 2N) real error vectors NEVER used in fitting.
    designs : iterable of (t_um (N,), n0 (N,)) calibration designs.

    Returns dict(moments=..., induced=..., per_design=[...]) where
    `induced` averages each distance over the bank.
    """
    designs = list(designs)
    if not designs:
        raise ValueError("need at least one calibration design")
    moments = moment_errors(x_model, x_heldout)
    per = []
    for t_um, n0 in designs:
        jm = induced_merits(x_model, t_um, n0, lam_um, shape, weights,
                            const, n_inc, n_sub, t_clip, n_clip)
        jr = induced_merits(x_heldout, t_um, n0, lam_um, shape,
                            weights, const, n_inc, n_sub, t_clip,
                            n_clip)
        per.append(distribution_distances(jm, jr, alpha))
    keys = ("d_mean", "d_P", "d_CVaR", "W1")
    induced = {k: float(np.mean([p[k] for p in per])) for k in keys}
    induced["alpha"] = float(alpha)
    return dict(moments=moments, induced=induced, per_design=per)
