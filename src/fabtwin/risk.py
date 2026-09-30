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

How sure is a number from K draws? (new in 0.8.0.) `yield_interval`
gives the exact (Clopper-Pearson) confidence interval of a pass
fraction; `cvar_interval` the large-sample standard error and interval
of a CVaR estimate; `cvar_difference` the same for the difference of
two designs' CVaRs, paired when both were scored with the same random
draws (which removes most of the noise they share). These intervals
cover the Monte-Carlo sampling error only: a twin that is wrong about
the machine stays wrong however many draws are taken.
"""
from __future__ import annotations

import numpy as np

from . import tmm

__all__ = ["cvar", "tail_statistics", "pass_fail", "yield_fraction",
           "evaluate_under_process", "yield_interval", "cvar_interval",
           "cvar_difference"]


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


def yield_interval(passed, confidence=0.95, n=None):
    """Exact confidence interval of a yield (new in 0.8.0).

    passed : boolean array, one entry per device (for example the
        output of `pass_fail`), or, with `n` given, the NUMBER k of
        devices that passed out of n.
    confidence : the confidence level, e.g. 0.95.

    Returns a dict with keys "yield" (k/n), "lo", "hi", "k", "n",
    "confidence". The interval is
    the Clopper-Pearson one (C. J. Clopper and E. S. Pearson,
    Biometrika 26, 404 (1934)): lo is the yield at which k or more
    passes would happen with probability (1 - confidence)/2, hi the
    yield at which k or fewer would; lo = 0 when k = 0 and hi = 1 when
    k = n. For any true yield the interval contains it with probability
    at least `confidence` (it is conservative: often more). It assumes
    independent devices with one common pass probability.
    """
    from scipy.stats import beta
    c = float(confidence)
    if not (0.0 < c < 1.0):
        raise ValueError("confidence must lie in (0, 1)")
    if n is None:
        ok = np.asarray(passed)
        if ok.dtype != bool:
            raise ValueError("passed must be a boolean array (or give "
                             "the count k with n=)")
        n_ = int(ok.size)
        k = int(ok.sum())
    else:
        k, n_ = passed, n
        if int(k) != k or int(n_) != n_:
            raise ValueError("k and n must be whole numbers")
        k, n_ = int(k), int(n_)
    if n_ < 1 or not (0 <= k <= n_):
        raise ValueError("need n >= 1 devices and 0 <= k <= n passes")
    a = 1.0 - c
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2.0, k, n_ - k + 1))
    hi = 1.0 if k == n_ else float(beta.ppf(1.0 - a / 2.0, k + 1, n_ - k))
    return {"yield": k / n_, "lo": lo, "hi": hi, "k": k, "n": n_,
            "confidence": c}


def _cvar_influence(values, alpha, min_tail):
    v = np.asarray(values, dtype=float).ravel()
    if v.size < 2 or not np.all(np.isfinite(v)):
        raise ValueError("need at least 2 finite merit values")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    q = max(1, int(np.ceil(alpha * v.size)))
    if q < int(min_tail):
        raise ValueError(
            f"only {q} value(s) in the worst {100 * alpha:g} %: the "
            f"large-sample standard error needs at least {int(min_tail)} "
            f"(min_tail); use at least {int(np.ceil(min_tail / alpha))} "
            "draws")
    tau = np.sort(v)[q - 1]
    return tau - np.maximum(tau - v, 0.0) / alpha


def cvar_interval(values, alpha, confidence=0.95, min_tail=20):
    """CVaR of a merit sample with its standard error and a
    large-sample confidence interval (new in 0.8.0).

    values : (K,) independent merit draws (e.g. the `samples` of
        `evaluate_under_process`); alpha : the tail fraction.

    The estimate is `cvar(values, alpha)`. Its standard error comes
    from the Rockafellar-Uryasev form CVaR = max over tau of
    tau - E[(tau - J)_+] / alpha: at the maximizing tau (the alpha
    quantile) the derivative in tau vanishes, so to first order the
    estimate is the sample mean of psi_k = tau - (tau - J_k)_+ / alpha,
    and

        se = std(psi) / sqrt(K),

    with tau the ceil(alpha K)-th smallest value. Hancock and Manistre
    (North American Actuarial Journal 9(2), 129 (2005)) develop and
    test a large-sample variance for this estimator; the formula here
    is derived as just stated and checked against simulation in the
    tests. The interval is CVaR +/- z se with z the normal quantile of
    the confidence level. It is a large-sample approximation that
    covers somewhat less than its level when few draws fall in the
    tail (the estimate is biased upward by a small amount at small K).
    The tests simulate a normally distributed merit (alpha = 0.1, 2000
    seeded samples each) and assert that a nominal 95 % interval
    covers the true CVaR in 91 % to 97 % of samples with 40 draws in
    the tail (observed: 93.2 %) and in 88 % to 96 % with 20 (observed:
    91.1 %). Refused when fewer than `min_tail` draws fall in the
    tail.

    Returns dict(CVaR, se, lo, hi, n_tail, confidence).
    """
    from scipy.stats import norm
    c = float(confidence)
    if not (0.0 < c < 1.0):
        raise ValueError("confidence must lie in (0, 1)")
    psi = _cvar_influence(values, alpha, min_tail)
    K = psi.size
    est = cvar(values, alpha)
    se = float(np.std(psi, ddof=1) / np.sqrt(K))
    z = float(norm.ppf(0.5 + c / 2.0))
    return dict(CVaR=est, se=se, lo=est - z * se, hi=est + z * se,
                n_tail=max(1, int(np.ceil(alpha * K))), confidence=c)


def cvar_difference(values, baseline, alpha, confidence=0.95,
                    paired=True, min_tail=20):
    """CVaR(values) - CVaR(baseline) with its standard error and a
    large-sample confidence interval (new in 0.8.0).

    paired=True (default) means draw k of both samples came from the
    same random numbers -- two designs scored with the same seed and
    the same number of draws, as in `evaluate_under_process` with a
    fixed generator. The standard error is then that of the mean of
    psi_k(values) - psi_k(baseline) (see `cvar_interval`), which
    removes the noise the two share; paired=False treats the samples
    as independent (se = sqrt(se_a^2 + se_b^2)). Pairing samples that
    were not drawn with common random numbers gives a wrong se.

    Returns dict(difference, se, lo, hi, confidence, paired).
    """
    from scipy.stats import norm
    c = float(confidence)
    if not (0.0 < c < 1.0):
        raise ValueError("confidence must lie in (0, 1)")
    a = np.asarray(values, dtype=float).ravel()
    b = np.asarray(baseline, dtype=float).ravel()
    pa = _cvar_influence(a, alpha, min_tail)
    pb = _cvar_influence(b, alpha, min_tail)
    if paired:
        if a.size != b.size:
            raise ValueError("paired samples must have the same length")
        se = float(np.std(pa - pb, ddof=1) / np.sqrt(a.size))
    else:
        se = float(np.sqrt(np.var(pa, ddof=1) / a.size
                           + np.var(pb, ddof=1) / b.size))
    d = cvar(a, alpha) - cvar(b, alpha)
    z = float(norm.ppf(0.5 + c / 2.0))
    return dict(difference=d, se=se, lo=d - z * se, hi=d + z * se,
                confidence=c, paired=bool(paired))


def evaluate_under_process(sampler, t_um, n0, lam_um, shape, weights,
                           const, K, n_inc=1.0, n_sub=1.0, alpha=0.05,
                           model=None):
    """Score a design by K fresh draws from `sampler(t, n, K)` ->
    (t_tilde (K,N), n_tilde (K,N)) -- the reference process, a twin,
    or anything else with that signature. Returns tail_statistics of
    the merit plus the merit samples. model= takes a
    `fabtwin.OpticalModel` (any angle, absorbing layers, nonlinear
    merit; weights and const are then ignored)."""
    tt, nt = sampler(t_um, n0, K)
    S = np.asarray(shape, dtype=float)
    if S.ndim == 1:
        S = np.broadcast_to(S[None, :], (len(t_um), S.size))
    J = np.empty(K)
    for k in range(K):
        if model is not None:
            from .merits import model_spectra
            R_, T_, A_ = model_spectra(model, lam_um, tt[k], nt[k], S,
                                       n_inc, n_sub)
            J[k] = model.merit(R_, T_, A_)[0]
            continue
        nlay = np.asarray(nt[k])[:, None] * S
        T = tmm.transmittance(lam_um, tt[k], nlay, n_inc, n_sub)
        J[k] = tmm.merit(T, weights, const)
    out = tail_statistics(J, alpha)
    out["samples"] = J
    return out
