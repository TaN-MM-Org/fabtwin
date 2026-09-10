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

Merit functions are deliberately *linear in T* (weights plus a
constant), so the exact adjoint of `fabtwin.adjoint` needs only the
weight vector; `notch_weights` and `bandpass_weights` build the two
sensor-front-end merits of the FabGAN-ID study (Mahim et al., IEEE
Sensors J., 2026) for any band layout, and any user-supplied weight
vector works identically.
"""
from __future__ import annotations

import numpy as np

__all__ = ["stack_BC", "stack_rt", "transmittance", "reflectance",
           "notch_weights", "bandpass_weights", "merit"]


def _phases_and_admittances(lam_um, t_um, n_layers, n_inc, n_sub,
                            theta0_rad, pol):
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
    if np.any(np.imag(n) < 0) or np.imag(complex(n_inc)) < 0 \
            or np.any(np.imag(np.asarray(n_sub)) < 0):
        raise ValueError("negative Im(n) (gain) is out of scope; "
                         "absorbing media carry n + i k with k >= 0")
    n = np.conj(n)
    n_inc = np.conj(complex(n_inc))
    n_sub_arr = np.conj(np.asarray(n_sub)) + 0.0j
    if n_sub_arr.ndim == 0:
        n_sub_arr = np.full(lam.size, n_sub_arr)
    if pol not in ("s", "p"):
        raise ValueError("pol must be 's' or 'p'")
    s0 = n_inc * np.sin(float(theta0_rad))
    # complex Snell cosines, principal branch
    cos_l = np.sqrt(1.0 - (s0 / n) ** 2 + 0.0j)
    cos_sub = np.sqrt(1.0 - (s0 / n_sub_arr) ** 2 + 0.0j)
    cos_0 = np.sqrt(1.0 - (s0 / n_inc) ** 2 + 0.0j)
    delta = 2.0 * np.pi * n * cos_l * t[:, None] / lam[None, :]
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
             theta0_rad=0.0, pol="s"):
    """The Macleod (B, C) vectors and the admittances (eta0, eta_sub).

    lam_um : (L,) wavelengths; t_um : (N,) thicknesses;
    n_layers : (N,) or (N, L), real or complex.
    Returns B, C, eta0 (complex, possibly (L,)), eta_sub (L,).
    """
    delta, eta, eta0, eta_sub = _phases_and_admittances(
        lam_um, t_um, n_layers, n_inc, n_sub, theta0_rad, pol)
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


def stack_rt(lam_um, t_um, n_layers, n_inc=1.0, n_sub=1.0,
             theta0_rad=0.0, pol="s"):
    """Reflectance and transmittance (R, T) of the stack."""
    B, C, eta0, eta_sub = stack_BC(lam_um, t_um, n_layers, n_inc, n_sub,
                                   theta0_rad, pol)
    denom = eta0 * B + C
    r = (eta0 * B - C) / denom
    R = np.abs(r) ** 2
    T = 4.0 * np.real(eta0) * np.real(eta_sub) / np.abs(denom) ** 2
    return R, T


def transmittance(lam_um, t_um, n_layers, n_inc=1.0, n_sub=1.0,
                  theta0_rad=0.0, pol="s"):
    """Intensity transmittance T(lam)."""
    return stack_rt(lam_um, t_um, n_layers, n_inc, n_sub,
                    theta0_rad, pol)[1]


def reflectance(lam_um, t_um, n_layers, n_inc=1.0, n_sub=1.0,
                theta0_rad=0.0, pol="s"):
    """Intensity reflectance R(lam)."""
    return stack_rt(lam_um, t_um, n_layers, n_inc, n_sub,
                    theta0_rad, pol)[0]


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


def merit(T, weights, const):
    """J = w . T + const for one spectrum or a batch (..., L)."""
    T = np.asarray(T)
    return T @ np.asarray(weights) + const
