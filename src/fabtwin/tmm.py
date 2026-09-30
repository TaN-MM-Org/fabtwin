"""Exact transfer-matrix optics for arbitrary stratified stacks.

Characteristic-matrix formulation (H. A. Macleod, Thin-Film Optical
Filters, 4th ed., 2010): for layers i = 1..N between an incidence
medium and a substrate,

    [B; C] = { prod_i [[cos d_i, i sin d_i / eta_i],
                       [i eta_i sin d_i, cos d_i]] } [1; eta_sub],

with phase thickness d_i = 2 pi n_i t_i cos(theta_i) / lam and tilted
admittances eta = n cos(theta) (s) or n / cos(theta) (p); Snell's law
n_inc sin(theta_0) = n_i sin(theta_i) continued into the complex plane
for absorbing layers. Reflectance and transmittance follow

    r = (eta_0 B - C) / (eta_0 B + C),   R = |r|^2,
    t = 2 eta_0 / (eta_0 B + C),         T = 4 eta_0 Re(eta_sub) / |eta_0 B + C|^2.

Everything here is closed-form linear algebra -- no fitting, no
surrogates -- and the tests hold it to closed forms rather than trust
it: the bare-interface Fresnel transmittance, the exact absentee
half-wave layer, the quarter-wave admittance transform, R + T = 1 for
lossless stacks at machine precision, the analytic Brewster zero of
p-polarized interface reflectance, the equality of s and p at normal
incidence, and agreement with the independent open-source `tmm`
reference (S. J. Byrnes, arXiv:1603.02720) on random stacks.

A thick substrate with a bare back face (new in 0.8.0, `n_exit=`):
the coherent coating above is combined with the substrate's back face
by summing the multiple reflections between them in power
(incoherently), which is how a spectrophotometer sees a coated plate
much thicker than the coherence length of its light; the tests hold
this to the incoherent solver `inc_tmm` of the same `tmm` reference
and to the closed-form bare plate T = 2n / (n^2 + 1).

The merit functions here are *linear in T* (weights plus a
constant), so the exact adjoint of `fabtwin.adjoint` needs only the
weight vector (nonlinear merits of R, T and A at any angle live in
`fabtwin.merits`, new in 0.7.0); `notch_weights` and `bandpass_weights` build the two
sensor-front-end merits of the FabGAN-ID study (Mahim et al., IEEE
Sensors J., 2026) for any band layout, and any user-supplied weight
vector works identically.
"""
from __future__ import annotations

import numpy as np

__all__ = ["stack_BC", "stack_rt", "transmittance", "reflectance",
           "notch_weights", "bandpass_weights", "merit",
           "weights_from_reflectance"]


def _cos_branch(n, s0):
    """cos(theta) in a medium of (conjugated) index n, with n cos(theta)
    on the physical (fourth-quadrant) branch."""
    cos = np.sqrt(1.0 - (s0 / n) ** 2 + 0.0j)      # principal root
    ncos = n * cos
    flip = (np.imag(ncos) > 0.0) & (np.abs(np.real(ncos))
                                    <= 1e-14 * np.abs(ncos))
    return np.where(flip, -cos, cos)


def _phases_and_admittances(lam_um, t_um, n_layers, n_inc, n_sub,
                            theta0_rad, pol, s0=None):
    # s0 (internal, new in 0.8.0): the Snell invariant n sin(theta),
    # given directly when the incidence medium is a dispersive (L,)
    # array -- the substrate seen from inside, for the thick-substrate
    # calculation. The public path (s0=None) is unchanged.
    lam = np.asarray(lam_um, dtype=float)
    t = np.asarray(t_um, dtype=float)
    n = np.asarray(n_layers)
    if n.ndim == 1:                     # dispersionless layers
        n = np.repeat(n[:, None], lam.size, axis=1)
    if n.shape != (t.size, lam.size):
        raise ValueError("n_layers must be (N,) or (N, L) matching "
                         "t_um and lam_um")
    # convention: n + i k with k >= 0 is ABSORBING (the common Python
    # ecosystem convention, e.g. the open tmm package); the Macleod
    # characteristic-matrix recursion below wants n - i k, so conjugate
    # internally. Gain media (k < 0) are refused, not extrapolated.
    if s0 is None:
        inc_gain = np.imag(complex(n_inc)) < 0
    else:
        inc_gain = np.any(np.imag(np.asarray(n_inc)) < 0)
    if np.any(np.imag(n) < 0) or inc_gain \
            or np.any(np.imag(np.asarray(n_sub)) < 0):
        raise ValueError("negative Im(n) (gain) is out of scope; "
                         "absorbing media carry n + i k with k >= 0")
    n = np.conj(n)
    if s0 is None:
        n_inc = np.conj(complex(n_inc))
    else:
        n_inc = np.conj(np.asarray(n_inc)) + 0.0j
    n_sub_arr = np.conj(np.asarray(n_sub)) + 0.0j
    if n_sub_arr.ndim == 0:
        n_sub_arr = np.full(lam.size, n_sub_arr)
    if pol not in ("s", "p"):
        raise ValueError("pol must be 's' or 'p' ('u' in stack_rt, "
                         "transmittance and reflectance)")
    if s0 is None:
        s0 = n_inc * np.sin(float(theta0_rad))
    # complex Snell cosines. In the conjugated (n - i k) convention a
    # wave travelling or decaying away from the interface has
    # n cos(theta) in the fourth quadrant (Re >= 0, Im <= 0). The
    # principal square root gives that automatically for absorbing
    # media, but for a LOSSLESS medium beyond the critical angle
    # (evanescent) it returns +i|.| on the branch cut; the physical
    # root is -i|.|. (0.6.1 and earlier took the principal root there,
    # which gave wrong R beyond the critical angle when the stack also
    # absorbed; see the 0.7.0 changelog.)
    cos_l = _cos_branch(n, s0)
    cos_sub = _cos_branch(n_sub_arr, s0)
    cos_0 = _cos_branch(n_inc, s0)
    delta = 2.0 * np.pi * n * cos_l * t[:, None] / lam[None, :]
    if pol == "p" and np.any(np.abs(cos_sub) == 0.0):
        raise ValueError("the substrate is exactly at its critical angle; "
                         "the p admittance n / cos(theta) is infinite "
                         "there")
    if pol == "s":
        eta = n * cos_l
        eta_sub = n_sub_arr * cos_sub
        eta0 = n_inc * cos_0
    else:
        eta = n / cos_l
        eta_sub = n_sub_arr / cos_sub
        eta0 = n_inc / cos_0
    return delta, eta, eta0, eta_sub


def stack_BC(lam_um, t_um, n_layers, n_inc=1.0, n_sub=1.0,
             theta0_rad=0.0, pol="s", _s0=None):
    """The Macleod (B, C) vectors and the admittances (eta0, eta_sub).

    lam_um : (L,) wavelengths; t_um : (N,) thicknesses;
    n_layers : (N,) or (N, L), real or complex.
    Returns B, C, eta0 (complex, possibly (L,)), eta_sub (L,).
    """
    delta, eta, eta0, eta_sub = _phases_and_admittances(
        lam_um, t_um, n_layers, n_inc, n_sub, theta0_rad, pol, _s0)
    c = np.cos(delta)
    s = np.sin(delta)
    B = np.ones(delta.shape[1], dtype=complex)
    C = eta_sub.astype(complex).copy()
    # right-to-left accumulation of M_1 ... M_N [1; eta_sub]
    for i in range(delta.shape[0] - 1, -1, -1):
        Bi = c[i] * B + 1j * s[i] / eta[i] * C
        Ci = 1j * eta[i] * s[i] * B + c[i] * C
        B, C = Bi, Ci
    return B, C, eta0, eta_sub


def _coherent_rt(lam_um, t_um, n_layers, n_inc, n_sub, theta0_rad, pol,
                 s0=None):
    B, C, eta0, eta_sub = stack_BC(lam_um, t_um, n_layers, n_inc, n_sub,
                                   theta0_rad, pol, s0)
    denom = eta0 * B + C
    r = (eta0 * B - C) / denom
    R = np.abs(r) ** 2
    T = 4.0 * np.real(eta0) * np.real(eta_sub) / np.abs(denom) ** 2
    return R, T


def _thick_substrate_media(lam_um, n_inc, n_sub, n_exit, theta0_rad):
    """Check and prepare the media of a thick-substrate calculation.

    Returns (n_inc float, n_sub (L,), n_exit (L,), s0) with
    s0 = n_inc sin(theta0), the Snell invariant. Refuses an absorbing
    incidence medium, substrate or exit medium (the incoherent power
    sum below is written for lossless ones), and an angle at which no
    light travels in the substrate (s0 >= n_sub: nothing reaches the
    back face, so it plays no role; use n_exit=None)."""
    lam = np.asarray(lam_um, dtype=float)
    L = lam.size
    out = []
    for name, v in (("n_inc", n_inc), ("n_sub", n_sub),
                    ("n_exit", n_exit)):
        a = np.asarray(v)
        if np.iscomplexobj(a) and np.any(np.imag(a) != 0):
            raise ValueError(
                f"with n_exit (a thick substrate with a bare back face), "
                f"{name} must be lossless (real): the back-face "
                "reflections are summed in power for a non-absorbing "
                "substrate")
        a = np.real(a).astype(float)
        if name == "n_inc":
            if a.ndim != 0:
                raise ValueError("n_inc must be a scalar with n_exit")
        elif a.ndim == 0:
            a = np.full(L, float(a))
        elif a.shape != (L,):
            raise ValueError(f"{name} must be a scalar or (L,)")
        if not np.all(np.isfinite(a)) or np.any(a <= 0):
            raise ValueError(f"{name} must be positive and finite")
        out.append(a)
    ni, ns, ne = out
    s0 = float(ni) * np.sin(float(theta0_rad))
    if np.any(s0 >= ns):
        raise ValueError(
            "no light travels in the substrate at this angle (beyond its "
            "critical angle), so its back face plays no role; use "
            "n_exit=None")
    return float(ni), ns, ne, s0


def _thick_rt(lam_um, t_um, n_layers, n_inc, n_sub, n_exit, theta0_rad,
              pol):
    lam = np.asarray(lam_um, dtype=float)
    ni, ns, ne, s0 = _thick_substrate_media(lam, n_inc, n_sub, n_exit,
                                           theta0_rad)
    t = np.asarray(t_um, dtype=float)
    n = np.asarray(n_layers)
    Rf, Tf = _coherent_rt(lam, t, n, ni, ns, theta0_rad, pol)
    # the same stack seen from inside the substrate (layers reversed)
    Rr, Tr = _coherent_rt(lam, t[::-1], n[::-1], ns, ni, 0.0, pol, s0)
    # the bare back face: substrate -> exit medium
    Rb, Tb = _coherent_rt(lam, np.zeros(0), np.zeros((0, lam.size)), ns,
                          ne, 0.0, pol, s0)
    den = 1.0 - Rr * Rb
    return Rf + Tf * Tr * Rb / den, Tf * Tb / den


def stack_rt(lam_um, t_um, n_layers, n_inc=1.0, n_sub=1.0,
             theta0_rad=0.0, pol="s", n_exit=None):
    """Reflectance and transmittance (R, T) of the stack. pol is "s",
    "p" or (new in 0.7.0) "u", unpolarized light: the mean of s and p.

    n_exit (new in 0.8.0) : None (default) treats the substrate as
    infinitely thick, as before: R and T are the powers reflected by
    the coating and transmitted INTO the substrate. A number (or an
    (L,) array) instead makes the substrate a thick, lossless slab
    whose bare back face meets a medium of index n_exit (1.0 for air):
    R and T are then the powers leaving the whole sample, which is what
    a spectrophotometer measures on a coated glass plate. The light
    bouncing between the coating and the back face is added in power,
    not in amplitude (incoherently), as for a substrate much thicker
    than the coherence length of the measuring light:

        T = Tf Tb / (1 - Rr Rb),   R = Rf + Tf Tr Rb / (1 - Rr Rb),

    with Rf, Tf the coating seen from the incidence side, Rr, Tr the
    coating seen from inside the substrate, and Rb, Tb the back face.
    Unpolarized light is summed per polarization first. The
    incidence medium, substrate and exit medium must be lossless.
    """
    if pol == "u":
        Rs, Ts = stack_rt(lam_um, t_um, n_layers, n_inc, n_sub,
                          theta0_rad, "s", n_exit)
        Rp, Tp = stack_rt(lam_um, t_um, n_layers, n_inc, n_sub,
                          theta0_rad, "p", n_exit)
        return 0.5 * (Rs + Rp), 0.5 * (Ts + Tp)
    if n_exit is not None:
        if pol not in ("s", "p"):
            raise ValueError("pol must be 's', 'p' or 'u'")
        return _thick_rt(lam_um, t_um, n_layers, n_inc, n_sub, n_exit,
                         theta0_rad, pol)
    return _coherent_rt(lam_um, t_um, n_layers, n_inc, n_sub, theta0_rad,
                        pol)


def transmittance(lam_um, t_um, n_layers, n_inc=1.0, n_sub=1.0,
                  theta0_rad=0.0, pol="s", n_exit=None):
    """Intensity transmittance T(lam) (n_exit: see `stack_rt`)."""
    return stack_rt(lam_um, t_um, n_layers, n_inc, n_sub,
                    theta0_rad, pol, n_exit)[1]


def reflectance(lam_um, t_um, n_layers, n_inc=1.0, n_sub=1.0,
                theta0_rad=0.0, pol="s", n_exit=None):
    """Intensity reflectance R(lam) (n_exit: see `stack_rt`)."""
    return stack_rt(lam_um, t_um, n_layers, n_inc, n_sub,
                    theta0_rad, pol, n_exit)[0]


# ------------------------- linear merits --------------------------


def notch_weights(lam_um, center_um, half_um, guard_um):
    """Weights (w, const) of the fluorescence-rejection notch merit

        J = 0.5 * mean_pass T + 0.5 * mean_stop (1 - T) = w . T + const

    with stop band [c-h, c+h] and pass bands outside the guard
    [c-h-g, c+h+g] (Mahim et al., IEEE Sensors J., 2026, Eq. 2).
    """
    lam = np.asarray(lam_um, dtype=float)
    stop = (lam >= center_um - half_um) & (lam <= center_um + half_um)
    pas = (lam <= center_um - half_um - guard_um) | \
          (lam >= center_um + half_um + guard_um)
    if not stop.any() or not pas.any():
        raise ValueError("band layout leaves an empty stop or pass band "
                         "on this wavelength grid")
    w = 0.5 * pas / pas.sum() - 0.5 * stop / stop.sum()
    return w, 0.5


def bandpass_weights(lam_um, lo_um, hi_um, guard_um):
    """Weights of the dual bandpass merit: J = 0.5 mean_in T
    + 0.5 mean_out (1 - T), pass band [lo, hi], rejection outside the
    guard."""
    lam = np.asarray(lam_um, dtype=float)
    inside = (lam >= lo_um) & (lam <= hi_um)
    outside = (lam <= lo_um - guard_um) | (lam >= hi_um + guard_um)
    if not inside.any() or not outside.any():
        raise ValueError("band layout leaves an empty band on this grid")
    w = 0.5 * inside / inside.sum() - 0.5 * outside / outside.sum()
    return w, 0.5


def weights_from_reflectance(weights_R, const=0.0):
    """Convert a reflectance-based linear merit into transmittance
    weights, so mirror and high-reflector merits run through the same
    exact adjoint (new in v0.3).

    For a LOSSLESS stack R = 1 - T exactly (asserted to machine
    precision in the solver tests), so

        J = w_R . R + c = (-w_R) . T + (c + sum w_R)

    is an identity, and the returned pair (w_T, const_T) feeds
    `merit`, `merit_and_grad`, `inverse_design` and `robustify`
    unchanged. For an ABSORBING stack R = 1 - T - A and the identity
    fails by exactly w_R . A -- which is why the lossless hand-adjoint
    regime is the scope here, matching `fabtwin.adjoint`.
    """
    w = np.asarray(weights_R, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights_R must be a 1-D weight vector")
    return -w, float(const) + float(w.sum())


def merit(T, weights, const):
    """J = w . T + const for one spectrum or a batch (..., L)."""
    T = np.asarray(T)
    return T @ np.asarray(weights) + const
