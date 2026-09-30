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

Scope, stated plainly: `errors_from_spectrum` fits thickness errors
only, indices held at the recipe values. Joint thickness-and-index
recovery from a single normal-incidence transmittance is the
classically unreliable problem the papers above dissect.
`errors_from_spectra` (new in 0.7.0) fits several measurements at
once (angles, polarizations, R and T) and can fit index errors too;
every refusal above still applies, so whether a measurement set
determines the indices is decided by the data, case by case -- the
tests show one normal-incidence spectrum refused and a multi-angle
set recovering both.

New in 0.8.0: `n_exit=` fits the spectrum of a whole coated plate,
back face included (see `fabtwin.tmm.stack_rt`); and both functions
report, and warn about, errors stopped by the search box (`at_bound`)
and fits inconsistent with the stated noise level (`chi2_pvalue`
below 1e-6) -- results that were returned silently before.
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as _chi2_dist

from .adjoint import transmittance_and_grads

__all__ = ["SpectrumRecovery", "errors_from_spectrum", "Measurement",
           "JointRecovery", "errors_from_spectra"]


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
    at_bound : (N,) bool (new in 0.8.0) -- True where the recovered
    error sits on the bound of the search box (+/- 2 max_error). Such a
    value is not a fitted answer, and its sigma means nothing: the true
    error is outside the box, or the model misses something in the
    measurement (for example the substrate's back face; see n_exit). A
    RuntimeWarning is issued whenever any entry is True.
    chi2_pvalue : (new in 0.8.0) probability that chi2 would be at
    least this large for a correct model with the stated sigma_T (the
    chi-square upper tail with dof degrees of freedom); None when
    sigma_T was not given or dof is 0. A tiny value means the model
    does not describe the measurement; below 1e-6 a RuntimeWarning is
    issued.
    """

    dt_over_t: np.ndarray
    t_um: np.ndarray
    sigma: np.ndarray
    chi2: float
    dof: int
    condition_number: float
    n_converged: int
    at_bound: np.ndarray = None
    chi2_pvalue: object = None


def _bound_check(x, active_mask, names):
    """at_bound mask from least_squares' active_mask, with a warning."""
    at = np.asarray(active_mask) != 0
    if np.any(at):
        warnings.warn(
            "recovered " + ", ".join(names[i] for i in np.nonzero(at)[0])
            + " sit(s) on the bound of the search box: that value is not "
            "a fitted answer and its uncertainty means nothing. The true "
            "error is larger than the box allows, or the model misses "
            "something in the measurement (for a coated plate measured in "
            "a spectrophotometer, the substrate's back face: pass n_exit)",
            RuntimeWarning, stacklevel=3)
    return at


CHI2_WARN_PVALUE = 1e-6


def _chi2_pvalue(chi2, dof, have_sigma):
    """Chi-square upper-tail probability, with a warning when the fit is
    statistically inconsistent with the stated noise level."""
    if not have_sigma or dof <= 0:
        return None
    p = float(_chi2_dist.sf(chi2, dof))
    if p < CHI2_WARN_PVALUE:
        warnings.warn(
            f"chi2 = {chi2:.4g} for {dof} degrees of freedom (p = {p:.3g}): "
            "the fitted model does not describe the measurement at the "
            "stated noise level, so the recovered errors and their "
            "uncertainties are not trustworthy. Check the noise level, the "
            "recipe, the substrate and, for a coated plate, its back face "
            "(n_exit)", RuntimeWarning, stacklevel=3)
    return p


def errors_from_spectrum(lam_um, T_meas, recipe_t_um, n0, shape,
                         n_inc=1.0, n_sub=1.0, sigma_T=None,
                         max_error=0.10, n_starts=8, seed=0,
                         chi2_rtol=1e-3, distinct_atol=1e-4,
                         n_exit=None):
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

    n_exit : None (default: an infinitely thick substrate) or the real
    index behind a thick, lossless substrate with a bare back face
    (1.0 for air; new in 0.8.0). A spectrophotometer measures the whole
    coated plate, back face included; fitting such a spectrum with
    n_exit=None asks the layers to explain the back-face loss (about
    4 % for glass in air) and gives wrong errors.

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
        sig = np.broadcast_to(np.asarray(sigma_T, dtype=float), (L,))
        if np.any(~np.isfinite(sig)) or np.any(sig <= 0):
            raise ValueError("sigma_T must be positive and finite")
        w = 1.0 / sig

    if n_exit is None:
        def T_and_dTdt(tt):
            T, dT_dt, _ = transmittance_and_grads(lam, tt, n0, shape,
                                                  n_inc=n_inc,
                                                  n_sub=n_sub)
            return T, dT_dt
    else:
        from .gradients import layer_indices, stack_rta_and_grads
        # the lossless-design checks of the adjoint path, kept
        transmittance_and_grads(lam, t0, n0, shape, n_inc=n_inc,
                                n_sub=n_sub)
        n_lay = layer_indices(np.asarray(n0, dtype=float), shape)

        def T_and_dTdt(tt):
            g = stack_rta_and_grads(lam, tt, n_lay, n_inc, n_sub, 0.0,
                                    "s", n_exit=n_exit)
            return g["T"], g["dT_dt"]

    def resid(x):
        return (T_and_dTdt(t0 * (1.0 + x))[0] - Tm) * w

    def jac(x):
        dT_dt = T_and_dTdt(t0 * (1.0 + x))[1]
        return (dT_dt * t0[:, None]).T * w[:, None]      # (L, N)

    resid(np.zeros(N))          # input errors surface here (0.7.0)
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
            sols.append((float(np.sum(r.fun ** 2)), r.x, r.active_mask))
    if not sols:
        raise ValueError("no multi-start converged; the spectrum is "
                         "inconsistent with the recipe within the "
                         "search box")
    sols.sort(key=lambda s: s[0])
    chi2_best, x_best, active = sols[0]
    for chi2_i, x_i, _ in sols[1:]:
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
    at = _bound_check(x_best, active,
                      [f"layer {i + 1} thickness error" for i in range(N)])
    return SpectrumRecovery(
        dt_over_t=x_best, t_um=t0 * (1.0 + x_best),
        sigma=np.sqrt(np.diag(cov)), chi2=chi2_best, dof=dof,
        condition_number=float(sv[0] / sv[-1]), n_converged=len(sols),
        at_bound=at,
        chi2_pvalue=_chi2_pvalue(chi2_best, dof, sigma_T is not None))


# ----------------------------------------------------------------------
# Several measurements, and index errors too (new in 0.7.0)


@dataclasses.dataclass(frozen=True)
class Measurement:
    """One measured spectrum on the common wavelength grid.

    theta0_rad : angle of incidence; pol : "s", "p" or "u";
    quantity : "T" or "R"; values : (L,) measured powers in [0, 1];
    sigma : scalar or (L,) one-standard-deviation noise (None: unit
    weights and a residual-scaled uncertainty, as in
    `errors_from_spectrum`).
    """

    theta0_rad: float
    pol: str
    quantity: str
    values: object
    sigma: object = None


@dataclasses.dataclass
class JointRecovery:
    """Recovered thickness (and index) errors with their reliability.

    dt_over_t : (N,) relative thickness errors; dn : (N,) absolute
    index errors at the reference wavelength (zero where not fitted);
    t_um, n0 : the recovered stack; sigma_t, sigma_n : linear
    (Jacobian) one-sigma uncertainties; boot_sigma_t, boot_sigma_n :
    the same from a parametric bootstrap (None unless requested);
    chi2, dof, condition_number, n_converged, chi2_pvalue as in
    `SpectrumRecovery`; at_bound_t, at_bound_n : (N,) bool, True where
    that error sits on the search bound (new in 0.8.0; a RuntimeWarning
    is issued, see `SpectrumRecovery.at_bound`).
    """

    dt_over_t: np.ndarray
    dn: np.ndarray
    t_um: np.ndarray
    n0: np.ndarray
    sigma_t: np.ndarray
    sigma_n: np.ndarray
    boot_sigma_t: object
    boot_sigma_n: object
    chi2: float
    dof: int
    condition_number: float
    n_converged: int
    at_bound_t: np.ndarray = None
    at_bound_n: np.ndarray = None
    chi2_pvalue: object = None


def errors_from_spectra(lam_um, measurements, recipe_t_um, n0, shape,
                        fit_index=False, kext=None, n_inc=1.0, n_sub=1.0,
                        max_error=0.10, max_index_error=0.10, n_starts=8,
                        seed=0, chi2_rtol=1e-3, distinct_atol=1e-4,
                        cond_max=1e10, n_boot=0, n_exit=None):
    """Per-layer thickness errors -- and, if asked, index errors -- from
    one or SEVERAL measured spectra (new in 0.7.0).

    Joint thickness-and-index recovery from one normal-incidence
    transmittance is the classically unreliable problem (module
    docstring). What changes the picture is more independent
    information: R and T, several angles, both polarizations. This
    function fits all given measurements at once through the exact
    oblique, absorbing solver with exact Jacobians (`fabtwin.gradients`)
    and keeps every refusal of `errors_from_spectrum` (too few points,
    a singular or ill-conditioned Jacobian, several equally good
    answers). Whether a given measurement set pins down the indices is
    decided by those checks, case by case -- not assumed.

    lam_um : (L,) the common wavelength grid; measurements : list of
    `Measurement`; recipe_t_um, n0, shape : the recipe (n0 real at the
    reference wavelength, shape real (L,) or (N, L)); kext : fixed
    extinction (None, (L,) or (N, L)).
    fit_index : False (thickness only), True (every layer), or a
        boolean (N,) mask of layers whose index is fitted.
    max_error, max_index_error : half-widths of the start boxes for
        relative thickness and absolute index errors (the fit is
        bounded by twice each).
    cond_max : refusal threshold on the condition number of the
        weighted Jacobian.
    n_boot : if > 0, also a parametric bootstrap: n_boot synthetic
        data sets (best-fit spectra plus Gaussian noise of the stated,
        or residual-estimated, size) are refitted from the best fit, and
        the spread of their results is reported next to the linear
        uncertainties.
    n_exit : None or the real index behind a thick, lossless substrate
        with a bare back face (new in 0.8.0; see
        `errors_from_spectrum`). Applies to every measurement.
    """
    from .gradients import layer_indices, stack_rta_and_grads
    lam = np.asarray(lam_um, dtype=float)
    t0 = np.asarray(recipe_t_um, dtype=float)
    n0 = np.asarray(n0, dtype=float)
    if lam.ndim != 1:
        raise ValueError("lam_um must be 1-D")
    if np.any(t0 <= 0):
        raise ValueError("recipe thicknesses must be positive")
    N, L = t0.size, lam.size
    meas = list(measurements)
    if not meas:
        raise ValueError("give at least one Measurement")
    S = np.asarray(shape, dtype=float)
    if S.ndim == 1:
        S = np.broadcast_to(S[None, :], (N, L))
    if fit_index is True:
        mask = np.ones(N, bool)
    elif fit_index is False or fit_index is None:
        mask = np.zeros(N, bool)
    else:
        mask = np.asarray(fit_index, bool)
        if mask.shape != (N,):
            raise ValueError("fit_index mask must have one entry per "
                             "layer")
    nf = int(mask.sum())
    P = N + nf
    Y, Wt = [], []
    if len({m.sigma is None for m in meas}) > 1:
        raise ValueError("give sigma for every measurement or for none "
                         "(mixing weighted and unit-weight data would "
                         "compare numbers in different units)")
    for m in meas:
        if m.quantity not in ("T", "R"):
            raise ValueError("Measurement.quantity must be 'T' or 'R'")
        if m.pol not in ("s", "p", "u"):
            raise ValueError("Measurement.pol must be 's', 'p' or 'u'")
        v = np.asarray(m.values, dtype=float)
        if v.shape != (L,) or not np.all(np.isfinite(v)) \
                or np.any(v < 0) or np.any(v > 1):
            raise ValueError("each measurement needs L finite values in "
                             "[0, 1]")
        if m.sigma is None:
            w = np.ones(L)
        else:
            w = 1.0 / np.broadcast_to(np.asarray(m.sigma, float), (L,))
            if np.any(~np.isfinite(w)) or np.any(w <= 0):
                raise ValueError("sigma must be positive and finite")
        Y.append(v)
        Wt.append(w)
    Y = np.concatenate(Y)
    Wt = np.concatenate(Wt)
    if Y.size < P:
        raise ValueError(
            f"{Y.size} measured values cannot determine {P} unknowns: "
            "the problem is underdetermined before noise even enters")
    have_sigma = all(m.sigma is not None for m in meas)

    def model(p):
        xt = p[:N]
        dn = np.zeros(N)
        dn[mask] = p[N:]
        n = layer_indices(n0 + dn, S, kext)
        t = t0 * (1.0 + xt)
        pred, jac = [], []
        for m in meas:
            g = stack_rta_and_grads(lam, t, n, n_inc, n_sub,
                                    m.theta0_rad, m.pol, n_exit=n_exit)
            q = m.quantity
            pred.append(g[q])
            Jt = (g[f"d{q}_dt"] * t0[:, None]).T             # (L, N)
            Jn = (g[f"d{q}_dn"] * S).T[:, mask]              # (L, nf)
            jac.append(np.hstack([Jt, Jn]))
        return np.concatenate(pred), np.vstack(jac)

    def resid(p):
        return (model(p)[0] - Y_fit[0]) * Wt

    def jac(p):
        return model(p)[1] * Wt[:, None]

    lo = np.concatenate([np.full(N, -2.0 * max_error),
                         np.full(nf, -2.0 * max_index_error)])
    hi = -lo

    def solve(p_init):
        return least_squares(resid, p_init, jac=jac, bounds=(lo, hi),
                             method="trf", xtol=1e-14, ftol=1e-14,
                             gtol=1e-12)

    Y_fit = [Y]
    model(np.zeros(P))          # input errors surface here, not as
    #                             "no multi-start converged" below
    rng = np.random.default_rng(seed)
    starts = [np.zeros(P)]
    for _ in range(int(n_starts) - 1):
        starts.append(np.concatenate([
            rng.uniform(-max_error, max_error, N),
            rng.uniform(-max_index_error, max_index_error, nf)]))
    sols = []
    for p_init in starts:
        try:
            r = solve(p_init)
        except Exception:
            continue
        if r.success or r.status > 0:
            sols.append((float(np.sum(r.fun ** 2)), r.x, r.active_mask))
    if not sols:
        raise ValueError("no multi-start converged; the spectra are "
                         "inconsistent with the recipe within the "
                         "search box")
    sols.sort(key=lambda s_: s_[0])
    chi2_best, p_best, active = sols[0]
    for chi2_i, p_i, _ in sols[1:]:
        if chi2_i <= chi2_best * (1.0 + chi2_rtol) + 1e-15 and \
                np.max(np.abs(p_i - p_best)) > distinct_atol:
            raise ValueError(
                "non-unique recovery: distinct error vectors reproduce "
                "the measured spectra equally well (Tikhonravov and "
                "Trubetskov, Appl. Opt. 51, 245 (2012)); add "
                "measurements (angles, R and T), or fix layers you trust")
    J = jac(p_best)
    sv = np.linalg.svd(J, compute_uv=False)
    cond = sv[0] / max(sv[-1], np.finfo(float).tiny)
    if sv[-1] == 0.0 or cond > cond_max:
        raise ValueError(
            f"ill-conditioned recovery (condition number {cond:.3g}): "
            "some combination of the unknowns barely changes the "
            "spectra, so the data cannot determine it; add "
            "measurements, or fit fewer unknowns")
    cov = np.linalg.inv(J.T @ J)
    dof = Y.size - P
    scale = 1.0
    if not have_sigma and dof > 0:
        scale = chi2_best / dof
        cov = cov * scale
    sd = np.sqrt(np.diag(cov))
    sig_n = np.zeros(N)
    sig_n[mask] = sd[N:]
    dn_best = np.zeros(N)
    dn_best[mask] = p_best[N:]
    at = _bound_check(
        p_best, active,
        [f"layer {i + 1} thickness error" for i in range(N)]
        + [f"layer {i + 1} index error" for i in np.nonzero(mask)[0]])
    at_n = np.zeros(N, bool)
    at_n[mask] = at[N:]

    boot_t = boot_n = None
    if int(n_boot) > 0:
        base = model(p_best)[0]
        noise_sd = np.sqrt(scale) / Wt
        reps = []
        for _ in range(int(n_boot)):
            Y_fit[0] = base + rng.normal(0.0, noise_sd)
            try:
                r = solve(p_best.copy())
            except Exception:
                continue
            if r.success or r.status > 0:
                reps.append(r.x)
        Y_fit[0] = Y
        if len(reps) < max(2, int(0.9 * int(n_boot))):
            raise ValueError(
                f"only {len(reps)} of {int(n_boot)} bootstrap refits "
                "converged; the bootstrap spread would not be "
                "trustworthy")
        reps = np.array(reps)
        bs = reps.std(axis=0, ddof=1)
        boot_t = bs[:N]
        boot_n = np.zeros(N)
        boot_n[mask] = bs[N:]
    return JointRecovery(
        dt_over_t=p_best[:N], dn=dn_best, t_um=t0 * (1.0 + p_best[:N]),
        n0=n0 + dn_best, sigma_t=sd[:N], sigma_n=sig_n,
        boot_sigma_t=boot_t, boot_sigma_n=boot_n, chi2=chi2_best,
        dof=dof, condition_number=float(cond), n_converged=len(sols),
        at_bound_t=at[:N], at_bound_n=at_n,
        chi2_pvalue=_chi2_pvalue(chi2_best, dof, have_sigma))
