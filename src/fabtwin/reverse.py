"""Reverse engineering: per-layer thickness errors from one measured
spectrum (new in v0.4).

`errors_from_traces` needs per-layer metrology. Most labs have
something cheaper: the recipe they sent to the machine and one
measured transmittance spectrum of what came back. This module
recovers the per-layer relative thickness errors from that spectrum
by weighted least squares through the exact transfer-matrix physics,
with the hand-derived adjoint supplying exact Jacobians and a
multi-start search guarding against the local minima a coating
merit landscape is full of.

Reliability is the hard part of reverse engineering, not optimization
-- different error vectors can reproduce a measured spectrum within
its noise, and an algorithm that silently returns one of them is a
random-number generator with good manners (A. V. Tikhonravov and
M. K. Trubetskov, Appl. Opt. 51, 245 (2012); T. V. Amotchkina,
M. K. Trubetskov, V. Pervak and A. V. Tikhonravov, Appl. Opt. 51,
5543 (2012)). This implementation therefore refuses rather than
guesses, three separate ways:

* fewer spectral points than layers is refused outright
  (underdetermined before any noise enters);
* an exactly or practically singular Jacobian at the solution is
  refused -- the exact case is real physics, not pathology: two
  ADJACENT layers of the same index enter the transfer matrix only
  through their thickness SUM, so their difference is invisible to
  any spectrum, and the tests pin this refusal with that exact
  degeneracy;
* multiple distinct error vectors found by the multi-start search
  fitting the spectrum equally well is refused as non-unique.

Scope, stated plainly: thickness errors only, indices held at the
recipe values. Joint thickness-and-index recovery from a single
normal-incidence transmittance is the classically unreliable problem
the papers above dissect; offering it casually would manufacture
false confidence, so it is designed out, not half-shipped.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.optimize import least_squares

from .adjoint import transmittance_and_grads

__all__ = ["SpectrumRecovery", "errors_from_spectrum"]


@dataclasses.dataclass
class SpectrumRecovery:
    """Recovered thickness errors and their reliability report.

    dt_over_t : (N,) relative thickness errors (x_i = t_i/recipe_i - 1).
    t_um : (N,) the recovered absolute thicknesses.
    sigma : (N,) one-standard-deviation uncertainties of dt_over_t
    (from the exact Jacobian; scaled by the residual when sigma_T was
    not given, exactly like a textbook least-squares fit).
    chi2, dof : weighted residual and degrees of freedom (chi2 is
    calibrated only when sigma_T reflects the real noise).
    condition_number : of the weighted Jacobian at the solution.
    n_converged : how many of the multi-starts converged; they all
    agreed, or this object would not exist.
    """

    dt_over_t: np.ndarray
    t_um: np.ndarray
    sigma: np.ndarray
    chi2: float
    dof: int
    condition_number: float
    n_converged: int


def errors_from_spectrum(lam_um, T_meas, recipe_t_um, n0, shape,
                         n_inc=1.0, n_sub=1.0, sigma_T=None,
                         max_error=0.10, n_starts=8, seed=0,
                         chi2_rtol=1e-3, distinct_atol=1e-4):
    """Per-layer relative thickness errors from one measured T(lam).

    lam_um, T_meas : the measured spectrum (transmittance in [0, 1]).
    recipe_t_um, n0, shape : the recipe as the adjoint module takes it
    (indices are FIXED at these values; module docstring says why).
    sigma_T : per-point (or scalar) one-sigma noise of T_meas; omitted
    means unit weights and a residual-scaled uncertainty.
    max_error : half-width of the fractional-error search box; starts
    are drawn inside it and the fit is bounded by 2x it.
    n_starts : multi-starts (first start is the recipe itself).
    chi2_rtol, distinct_atol : two solutions are "equally good" when
    their chi2 agree within chi2_rtol, and "distinct" when any layer's
    recovered error differs by more than distinct_atol -- both
    conditions together trigger the non-uniqueness refusal.

    Returns a `SpectrumRecovery`. Raises ValueError when the recovery
    is underdetermined, degenerate, or non-unique (module docstring).
    """
    lam = np.asarray(lam_um, dtype=float)
    Tm = np.asarray(T_meas, dtype=float)
    t0 = np.asarray(recipe_t_um, dtype=float)
    if lam.ndim != 1 or Tm.shape != lam.shape:
        raise ValueError("lam_um and T_meas must be matching 1D arrays")
    if np.any(t0 <= 0):
        raise ValueError("recipe thicknesses must be positive")
    N, L = t0.size, lam.size
    if L < N:
        raise ValueError(
            f"{L} spectral points cannot determine {N} layer errors: "
            "the problem is underdetermined before noise even enters; "
            "measure more wavelengths or fix layers")
    if np.any(Tm < 0) or np.any(Tm > 1) or not np.all(np.isfinite(Tm)):
        raise ValueError("T_meas must be finite transmittances in [0, 1]")
    if sigma_T is None:
        w = np.ones(L)
    else:
        w = 1.0 / np.broadcast_to(np.asarray(sigma_T, dtype=float), (L,))
        if np.any(~np.isfinite(w)) or np.any(w <= 0):
            raise ValueError("sigma_T must be positive and finite")

    def resid(x):
        T, _, _ = transmittance_and_grads(lam, t0 * (1.0 + x), n0, shape,
                                          n_inc=n_inc, n_sub=n_sub)
        return (T - Tm) * w

    def jac(x):
        _, dT_dt, _ = transmittance_and_grads(lam, t0 * (1.0 + x), n0,
                                              shape, n_inc=n_inc,
                                              n_sub=n_sub)
        return (dT_dt * t0[:, None]).T * w[:, None]      # (L, N)

    rng = np.random.default_rng(seed)
    bound = 2.0 * float(max_error)
    starts = [np.zeros(N)]
    starts += [rng.uniform(-max_error, max_error, N)
               for _ in range(int(n_starts) - 1)]
    sols = []
    for x_init in starts:
        try:
            r = least_squares(resid, x_init, jac=jac,
                              bounds=(-bound, bound), method="trf",
                              xtol=1e-14, ftol=1e-14, gtol=1e-12)
        except Exception:
            continue
        if r.success or r.status > 0:
            sols.append((float(np.sum(r.fun ** 2)), r.x))
    if not sols:
        raise ValueError("no multi-start converged; the spectrum is "
                         "inconsistent with the recipe within the "
                         "search box")
    sols.sort(key=lambda s: s[0])
    chi2_best, x_best = sols[0]
    for chi2_i, x_i in sols[1:]:
        close_fit = chi2_i <= chi2_best * (1.0 + chi2_rtol) + 1e-15
        distinct = np.max(np.abs(x_i - x_best)) > distinct_atol
        if close_fit and distinct:
            raise ValueError(
                "non-unique recovery: distinct error vectors reproduce "
                "the measured spectrum equally well (reverse-engineering "
                "reliability, Tikhonravov and Trubetskov, Appl. Opt. 51, "
                "245 (2012)); widen the spectral range, add points, or "
                "fix layers you trust")
    J = jac(x_best)
    sv = np.linalg.svd(J, compute_uv=False)
    if sv[-1] == 0.0 or sv[0] / max(sv[-1], np.finfo(float).tiny) > 1e10:
        raise ValueError(
            "singular recovery: some combination of layer errors leaves "
            "the spectrum unchanged (e.g. adjacent layers of the same "
            "index enter only through their thickness sum), so no "
            "spectrum can determine it; merge or fix the degenerate "
            "layers")
    cov = np.linalg.inv(J.T @ J)
    dof = L - N
    if sigma_T is None and dof > 0:
        cov = cov * (chi2_best / dof)            # residual-scaled
    return SpectrumRecovery(
        dt_over_t=x_best, t_um=t0 * (1.0 + x_best),
        sigma=np.sqrt(np.diag(cov)), chi2=chi2_best, dof=dof,
        condition_number=float(sv[0] / sv[-1]), n_converged=len(sols))
