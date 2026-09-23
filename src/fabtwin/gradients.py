"""Exact gradients beyond normal incidence and lossless layers (new in
0.7.0).

`fabtwin.adjoint` differentiates the normal-incidence transmittance of
a lossless stack. This module differentiates the full transfer-matrix
solver of `fabtwin.tmm`: any angle of incidence, s, p or unpolarized
light, absorbing layers (index n + i k, k >= 0), and all three powers,
reflectance R, transmittance T and absorptance A = 1 - R - T.

Method (the same prefix/suffix adjoint as `fabtwin.adjoint`, one layer
matrix at a time). The solver works with the conjugated index
n' = n - i k (Macleod's convention). Every quantity it builds from n'
-- the tilted phase delta = 2 pi q t / lam with q = n' cos(theta) =
sqrt(n'^2 - s0^2), the admittance eta = q (s) or n'^2 / q (p), the
layer matrices, B, C, the denominator D = eta0 B + C and the
amplitude r = (eta0 B - C) / D -- is a complex-analytic function of
n', so one complex derivative per layer and wavelength carries all the
information:

    dq/dn'      = n' / q
    d(eta_s)/dn' = n' / q,      d(eta_p)/dn' = n' (n'^2 - 2 s0^2) / q^3
    d(delta)/dn' = 2 pi t n' / (lam q),   d(delta)/dt = 2 pi q / lam.

With dD and dr from the prefix/suffix sweeps, and n = a + i k (so
n' = a - i k),

    dT/da = -2 T Re(D'/D),        dT/dk = -2 T Im(D'/D),
    dR/da = 2 Re(conj(r) r'),     dR/dk = 2 Im(conj(r) r'),
    dT/dt = -2 T Re(D_t/D),       dR/dt = 2 Re(conj(r) r_t),

and dA = -dR - dT. Unpolarized light is the average of s and p.

What the tests hold this module to: it equals `fabtwin.adjoint` at
normal incidence on lossless stacks; every derivative matches central
finite differences of the independent forward solver `fabtwin.tmm` on
random absorbing stacks at oblique incidence in both polarizations,
also beyond the critical angle; and (with the [twin] extra) it matches
automatic differentiation of a separate JAX implementation.

Scope, stated plainly: the incidence medium must be lossless (real
index), as for any angle-resolved power measurement; a layer whose
q = n' cos(theta) is (numerically) zero -- exactly at the critical
angle of a lossless layer -- has no derivative there and is refused.
"""
from __future__ import annotations

import numpy as np

from .tmm import _cos_branch

__all__ = ["stack_rta_and_grads", "layer_indices"]


def layer_indices(n0, shape, kext=None):
    """The complex layer indices n_i(lam) = n0_i S_i(lam) + i K_i(lam).

    n0 : (N,) real indices at the reference wavelength; shape : (L,)
    shared or (N, L) per-layer real dispersion shape; kext : None, (L,)
    or (N, L) extinction coefficients (>= 0), held fixed.
    Returns an (N, L) array (complex when kext is given).
    """
    n0 = np.asarray(n0, dtype=float)
    S = np.asarray(shape)
    if np.iscomplexobj(S):
        raise ValueError("the dispersion shape must be real; give the "
                         "absorption through kext")
    S = S.astype(float)
    if S.ndim == 1:
        S = np.broadcast_to(S[None, :], (n0.size, S.size))
    if S.shape[0] != n0.size:
        raise ValueError("per-layer shape must be (N, L)")
    n = n0[:, None] * S
    if kext is None:
        return n
    K = np.asarray(kext, dtype=float)
    if K.ndim == 1:
        if K.shape[0] != n.shape[1]:
            raise ValueError("a 1-D kext is k(lam), shared by all layers, "
                             "and needs one value per wavelength; give "
                             "per-layer values as (N, L) or (N, 1)")
        K = K[None, :]
    try:
        K = np.broadcast_to(K, n.shape)
    except ValueError:
        raise ValueError("kext must be (L,), (N, L) or (N, 1)") from None
    if np.any(K < 0) or not np.all(np.isfinite(K)):
        raise ValueError("kext must be finite and >= 0 (absorption)")
    return n + 1j * K


def _one_pol(lam, t, n_layers, n_inc, n_sub, theta0, pol):
    nprime = np.conj(np.asarray(n_layers, dtype=complex))   # (N, L)
    N, L = nprime.shape
    s0 = n_inc * np.sin(theta0)
    cos_l = _cos_branch(nprime, s0)
    q = nprime * cos_l                                       # n' cos
    if N and np.any(np.abs(q) < 1e-9 * np.abs(nprime)):
        raise ValueError("a layer sits exactly at its critical angle "
                         "(n cos(theta) = 0); the derivative does not "
                         "exist there")
    nsub = np.conj(np.asarray(n_sub, dtype=complex))
    if nsub.ndim == 0:
        nsub = np.full(L, nsub)
    cos_s = _cos_branch(nsub, s0)
    if pol == "p" and np.any(np.abs(cos_s) == 0.0):
        raise ValueError("the substrate is exactly at its critical angle; "
                         "the p admittance is infinite there")
    cos_0 = _cos_branch(np.complex128(n_inc), s0)
    if pol == "s":
        eta = q
        deta = nprime / q
        eta_sub = nsub * cos_s
        eta0 = n_inc * cos_0
    else:
        eta = nprime * nprime / q
        deta = nprime * (nprime * nprime - 2.0 * s0 * s0) / q ** 3
        eta_sub = nsub / cos_s
        eta0 = n_inc / cos_0
    two_pi_lam = 2.0 * np.pi / lam[None, :]
    delta = two_pi_lam * q * t[:, None]
    ddelta_dn = two_pi_lam * t[:, None] * nprime / q
    ddelta_dt = two_pi_lam * q
    c = np.cos(delta)
    s = np.sin(delta)

    # prefix matrices P_i = M_1 ... M_{i-1} (2x2 per wavelength)
    P = np.empty((N, 2, 2, L), dtype=complex)
    A = np.zeros((2, 2, L), dtype=complex)
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    for i in range(N):
        P[i] = A
        m00, m01 = c[i], 1j * s[i] / eta[i]
        m10, m11 = 1j * eta[i] * s[i], c[i]
        A = np.array([[A[0, 0] * m00 + A[0, 1] * m10,
                       A[0, 0] * m01 + A[0, 1] * m11],
                      [A[1, 0] * m00 + A[1, 1] * m10,
                       A[1, 0] * m01 + A[1, 1] * m11]])
    B = A[0, 0] + A[0, 1] * eta_sub
    C = A[1, 0] + A[1, 1] * eta_sub
    # suffix vectors r_i = M_{i+1} ... M_N [1; eta_sub]
    R0 = np.empty((N, L), dtype=complex)
    R1 = np.empty((N, L), dtype=complex)
    b0 = np.ones(L, dtype=complex)
    b1 = eta_sub.astype(complex)
    for i in range(N - 1, -1, -1):
        R0[i], R1[i] = b0, b1
        nb0 = c[i] * b0 + (1j * s[i] / eta[i]) * b1
        nb1 = (1j * eta[i] * s[i]) * b0 + c[i] * b1
        b0, b1 = nb0, nb1

    D = eta0 * B + C
    r = (eta0 * B - C) / D
    Rr = np.abs(r) ** 2
    T = 4.0 * np.real(eta0) * np.real(eta_sub) / np.abs(D) ** 2

    # dM/d(delta) and dM/d(eta) applied to the suffix vector
    dd0 = -s * R0 + 1j * c / eta * R1
    dd1 = 1j * eta * c * R0 - s * R1
    de0 = -1j * s / eta ** 2 * R1
    de1 = 1j * s * R0

    def prefix_apply(v0, v1):
        dB = P[:, 0, 0] * v0 + P[:, 0, 1] * v1
        dC = P[:, 1, 0] * v0 + P[:, 1, 1] * v1
        return dB, dC

    dB_dl, dC_dl = prefix_apply(dd0, dd1)           # per unit delta
    dB_de, dC_de = prefix_apply(de0, de1)           # per unit eta
    dB_n = dB_dl * ddelta_dn + dB_de * deta
    dC_n = dC_dl * ddelta_dn + dC_de * deta
    dB_t = dB_dl * ddelta_dt
    dC_t = dC_dl * ddelta_dt

    def powers(dB, dC):
        dD = eta0 * dB + dC
        dNum = eta0 * dB - dC
        dr = (dNum - r * dD) / D
        w = dD / D
        z = np.conj(r) * dr
        return w, z

    w_n, z_n = powers(dB_n, dC_n)
    w_t, z_t = powers(dB_t, dC_t)
    out = dict(R=Rr, T=T,
               dT_dt=np.real(-2.0 * T * w_t), dR_dt=2.0 * np.real(z_t),
               dT_dn=np.real(-2.0 * T * w_n), dR_dn=2.0 * np.real(z_n),
               dT_dk=-2.0 * T * np.imag(w_n), dR_dk=2.0 * np.imag(z_n))
    return out


def stack_rta_and_grads(lam_um, t_um, n_layers, n_inc=1.0, n_sub=1.0,
                        theta0_rad=0.0, pol="s"):
    """R, T, A and their exact derivatives for any stack and angle.

    lam_um : (L,) wavelengths; t_um : (N,) thicknesses; n_layers :
    (N, L) or (N,) indices n + i k (k >= 0); n_inc : real index of the
    incidence medium; n_sub : substrate index (scalar or (L,), may
    absorb); theta0_rad : angle of incidence; pol : "s", "p" or "u"
    (unpolarized: the mean of s and p).

    Returns a dict with R, T, A (L,) and, each (N, L), the derivatives
    with respect to every layer thickness (dR_dt, dT_dt, dA_dt), the
    real part of every layer index (dR_dn, dT_dn, dA_dn) and its
    extinction coefficient (dR_dk, dT_dk, dA_dk), at each wavelength.
    """
    lam = np.asarray(lam_um, dtype=float)
    t = np.asarray(t_um, dtype=float)
    if lam.ndim != 1 or t.ndim != 1:
        raise ValueError("lam_um and t_um must be 1-D")
    n = np.asarray(n_layers)
    if n.ndim == 1:
        n = np.repeat(n[:, None], lam.size, axis=1)
    if n.shape != (t.size, lam.size):
        raise ValueError("n_layers must be (N,) or (N, L)")
    if np.any(np.imag(n) < 0) or np.any(np.imag(np.asarray(n_sub)) < 0):
        raise ValueError("negative Im(n) (gain) is out of scope; "
                         "absorbing media carry n + i k with k >= 0")
    if np.iscomplexobj(np.asarray(n_inc)) and np.imag(n_inc) != 0:
        raise ValueError("the incidence medium must be lossless (real "
                         "index) for angle-resolved powers")
    n_inc = float(np.real(n_inc))
    if not (n_inc > 0):
        raise ValueError("n_inc must be positive")
    th = float(theta0_rad)
    if not (0.0 <= th < np.pi / 2):
        raise ValueError("theta0_rad must lie in [0, pi/2)")
    if pol in ("s", "p"):
        out = _one_pol(lam, t, n, n_inc, n_sub, th, pol)
    elif pol == "u":
        a = _one_pol(lam, t, n, n_inc, n_sub, th, "s")
        b = _one_pol(lam, t, n, n_inc, n_sub, th, "p")
        out = {k: 0.5 * (a[k] + b[k]) for k in a}
    else:
        raise ValueError("pol must be 's', 'p' or 'u'")
    out["A"] = 1.0 - out["R"] - out["T"]
    for p in ("t", "n", "k"):
        out[f"dA_d{p}"] = -out[f"dR_d{p}"] - out[f"dT_d{p}"]
    return out
