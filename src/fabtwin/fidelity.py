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
           "induced_merits", "twin_fidelity_report", "energy_distance",
           "drift_test", "novelty_pvalues"]


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
                   n_inc=1.0, n_sub=1.0, t_clip=None, n_clip=None,
                   model=None):
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
        if model is not None:
            from .merits import model_spectra
            R_, T_, A_ = model_spectra(model, lam_um, tt[k], nn[k], S,
                                       n_inc, n_sub)
            J[k] = model.merit(R_, T_, A_)[0]
            continue
        T = tmm.transmittance(lam_um, tt[k], nn[k][:, None] * S,
                              n_inc, n_sub)
        J[k] = tmm.merit(T, weights, const)
    return J


def twin_fidelity_report(x_model, x_heldout, designs, lam_um, shape,
                         weights, const, n_inc=1.0, n_sub=1.0,
                         alpha=0.05, t_clip=None, n_clip=None,
                         model=None):
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
                            const, n_inc, n_sub, t_clip, n_clip, model)
        jr = induced_merits(x_heldout, t_um, n0, lam_um, shape,
                            weights, const, n_inc, n_sub, t_clip,
                            n_clip, model)
        per.append(distribution_distances(jm, jr, alpha))
    keys = ("d_mean", "d_P", "d_CVaR", "W1")
    induced = {k: float(np.mean([p[k] for p in per])) for k in keys}
    induced["alpha"] = float(alpha)
    return dict(moments=moments, induced=induced, per_design=per)


# ----------------------------------------------------------------------
# Has the machine changed? Is this run unlike anything logged? (0.7.0)


def energy_distance(a, b):
    """Two-sample energy distance between samples a (K, D) and b (M, D):

        E = 2 mean|a - b| - mean|a - a'| - mean|b - b'|

    (Euclidean norms over all pairs; G. J. Szekely and M. L. Rizzo,
    "Testing for equal distributions in high dimension", InterStat
    (2004)). It is zero for identical samples and, in the population,
    zero only when the two distributions are equal."""
    a = np.atleast_2d(np.asarray(a, dtype=float))
    b = np.atleast_2d(np.asarray(b, dtype=float))
    if a.shape[1] != b.shape[1]:
        raise ValueError("samples must share the dimension D")

    from scipy.spatial.distance import cdist

    def mean_dist(u, v):
        return float(cdist(u, v).mean())

    return 2.0 * mean_dist(a, b) - mean_dist(a, a) - mean_dist(b, b)


def drift_test(x_old, x_new, n_perm=999, seed=0, standardize=True):
    """Permutation test of "the new runs come from the same process as
    the old ones", on error vectors (from `errors_from_traces`).

    The statistic is the energy distance; its null distribution is
    built by reshuffling the pooled runs n_perm times, and the p-value
    (1 + #{permuted >= observed}) / (1 + n_perm) is valid at every
    sample size when the runs are exchangeable under the null (the
    standard permutation argument). standardize=True scales every
    component by the pooled standard deviation first, so thickness and
    index errors count alike.

    Returns dict(statistic, p_value, n_old, n_new). A small p-value is
    evidence that the machine has drifted and the twin should be
    refitted; a large one is NOT proof that nothing changed (the test
    may lack power with few runs).
    """
    a = np.asarray(x_old, dtype=float)
    b = np.asarray(x_new, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("x_old and x_new must be (K, D) and (M, D)")
    if a.shape[0] < 2 or b.shape[0] < 2:
        raise ValueError("need at least 2 runs in each group")
    n_perm = int(n_perm)
    if n_perm < 19:
        raise ValueError("n_perm must be >= 19 (p-values below 0.05 "
                         "need at least 19 permutations)")
    pool = np.vstack([a, b])
    if standardize:
        sd = pool.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        pool = (pool - pool.mean(axis=0)) / sd
    K = a.shape[0]
    obs = energy_distance(pool[:K], pool[K:])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(pool.shape[0])
        if energy_distance(pool[p[:K]], pool[p[K:]]) >= obs:
            count += 1
    return dict(statistic=float(obs),
                p_value=(1.0 + count) / (1.0 + n_perm),
                n_old=int(K), n_new=int(b.shape[0]))


def novelty_pvalues(x_train, x_calib, x_new):
    """Conformal p-values of "this run is like the logged ones".

    A Gaussian twin is fitted to x_train; every run is scored by its
    Mahalanobis distance from that fit; a new run's p-value is

        p = (1 + #{calibration scores >= its score}) / (n_calib + 1).

    If the new run is exchangeable with the calibration runs,
    P(p <= u) <= u for every u (split-conformal outlier detection:
    Vovk, Gammerman and Shafer, Algorithmic Learning in a Random World
    (2005)). The score uses the twin only to rank runs; the validity
    does not depend on the twin being right. A small p-value flags a
    run whose error pattern the logged runs do not cover -- the case a
    twin cannot vouch for.
    """
    from .twins import GaussianTwin
    tr = np.asarray(x_train, dtype=float)
    cal = np.asarray(x_calib, dtype=float)
    new = np.atleast_2d(np.asarray(x_new, dtype=float))
    if tr.ndim != 2 or cal.ndim != 2 or cal.shape[0] < 1 \
            or new.shape[1] != cal.shape[1] or tr.shape[1] != cal.shape[1]:
        raise ValueError("x_train, x_calib, x_new must share D")
    tw = GaussianTwin(tr)

    def score(x):
        d = np.linalg.solve(tw.L, (x - tw.mu).T)
        return np.sqrt(np.sum(d * d, axis=0))

    sc = score(cal)
    sn = score(new)
    return (1.0 + np.sum(sc[None, :] >= sn[:, None], axis=1)) \
        / (sc.size + 1.0)
