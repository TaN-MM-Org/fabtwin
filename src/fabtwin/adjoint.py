"""Hand-derived exact adjoint of the transfer-matrix recursion.

The FabGAN-ID loop rests on exact design gradients through the
physics; the paper obtained them by reverse-mode autodiff in JAX.
This module derives the same discrete adjoint by hand, in NumPy, so
the fully differentiable fabrication loop needs no autodiff framework
at all. With u = [eta0, 1], v = [1; eta_sub] and M = M_1 ... M_N, the
denominator D = u M v determines T = 4 eta0 Re(eta_sub)/|D|^2, so

    dT/dp = -2 T Re[(dD/dp) / D],
    dD/dp_i = l_i (dM_i/dp) r_i,

with the prefix/suffix vectors l_i = u M_1..M_{i-1} and
r_i = M_{i+1}..M_N v accumulated in two O(N L) sweeps, and

    dM/d(delta) = [[-sin d, i cos d / eta], [i eta cos d, -sin d]],
    dM/d(eta)   = [[0, -i sin d / eta^2], [i sin d, 0]].

Layer i has n_i(lam) = n0_i S_i(lam) (real; `shape` is one shared
(L,) dispersion shape or a per-layer (N, L) array, so single-material
variable-index platforms and classic multi-material stacks use the
same adjoint) and phase delta_i = 2 pi n_i t_i / lam, so the chain to
the design parameters (t_i, n0_i) is elementary. Scope, stated
plainly: normal incidence and real (lossless) layer indices -- the
regime of the FabGAN-ID loop; the oblique and absorbing *forward*
solver lives in `fabtwin.tmm`, and its gradients are deliberately not
faked here.

Anchors asserted in the tests rather than stated: agreement with
central finite differences to a median relative error near 1e-10 (the
paper's own figure for the JAX adjoint), an exactly vanishing
gradient at the closed-form quarter-wave antireflection optimum, and
(when JAX is installed) agreement with autodiff through the twin
extra's solver.
"""
from __future__ import annotations

import numpy as np

__all__ = ["merit_and_grad", "transmittance_and_grads"]


def _layer_data(lam_um, t_um, n0, shape, n_sub):
    if np.iscomplexobj(np.asarray(t_um)) or np.iscomplexobj(np.asarray(n0)):
        raise ValueError("the hand adjoint covers real (lossless) "
                         "designs; use fabtwin.tmm for absorbing stacks")
    lam = np.asarray(lam_um, dtype=float)
    t = np.asarray(t_um, dtype=float)
    n0 = np.asarray(n0, dtype=float)
    S = np.asarray(shape, dtype=float)
    if S.ndim == 1:
        if S.shape != lam.shape:
            raise ValueError("shape must be S(lam) on the wavelength "
                             "grid, or (N, L) for per-layer dispersion")
        S = np.broadcast_to(S[None, :], (t.size, lam.size))
    elif S.shape != (t.size, lam.size):
        raise ValueError("per-layer shape must be (N, L)")
    n = n0[:, None] * S                          # (N, L) real
    delta = 2.0 * np.pi * n * t[:, None] / lam[None, :]
    eta = n
    eta_sub = np.asarray(n_sub, dtype=float)
    if eta_sub.ndim == 0:
        eta_sub = np.full(lam.size, float(eta_sub))
    return lam, t, n0, S, n, delta, eta, eta_sub


def transmittance_and_grads(lam_um, t_um, n0, shape, n_inc=1.0, n_sub=1.0):
    """T(lam) and its exact gradients dT/dt_i, dT/dn0_i.

    Returns (T (L,), dT_dt (N, L), dT_dn0 (N, L)).
    """
    lam, t, n0v, S, n, delta, eta, eta_sub = _layer_data(
        lam_um, t_um, n0, shape, n_sub)
    N, L = delta.shape
    eta0 = float(n_inc)
    c = np.cos(delta)
    s = np.sin(delta)

    # prefix l_i = u M_1..M_{i-1}; u = [eta0, 1]
    l0 = np.empty((N, L), dtype=complex)
    l1 = np.empty((N, L), dtype=complex)
    a0 = np.full(L, eta0, dtype=complex)
    a1 = np.ones(L, dtype=complex)
    for i in range(N):
        l0[i], l1[i] = a0, a1
        b0 = a0 * c[i] + a1 * (1j * eta[i] * s[i])
        b1 = a0 * (1j * s[i] / eta[i]) + a1 * c[i]
        a0, a1 = b0, b1
    D = a0 + a1 * eta_sub                        # u M v

    # suffix r_i = M_{i+1}..M_N v; v = [1, eta_sub]
    r0 = np.empty((N, L), dtype=complex)
    r1 = np.empty((N, L), dtype=complex)
    b0 = np.ones(L, dtype=complex)
    b1 = eta_sub.astype(complex).copy()
    for i in range(N - 1, -1, -1):
        r0[i], r1[i] = b0, b1
        nb0 = c[i] * b0 + (1j * s[i] / eta[i]) * b1
        nb1 = (1j * eta[i] * s[i]) * b0 + c[i] * b1
        b0, b1 = nb0, nb1

    T = 4.0 * eta0 * eta_sub / np.abs(D) ** 2

    dD_ddelta = (-s * (l0 * r0 + l1 * r1)
                 + 1j * c * (l0 * r1 / eta + eta * l1 * r0))
    dD_deta = 1j * s * (l1 * r0 - l0 * r1 / eta ** 2)
    dT_ddelta = -2.0 * T[None, :] * np.real(dD_ddelta / D[None, :])
    dT_deta = -2.0 * T[None, :] * np.real(dD_deta / D[None, :])

    two_pi_over_lam = 2.0 * np.pi / lam[None, :]
    dT_dt = dT_ddelta * n * two_pi_over_lam
    dT_dn0 = (dT_ddelta * t[:, None] * S * two_pi_over_lam
              + dT_deta * S)
    return T, dT_dt, dT_dn0


def merit_and_grad(lam_um, t_um, n0, shape, weights, const,
                   n_inc=1.0, n_sub=1.0):
    """J = w . T + const and its exact gradients (dJ/dt, dJ/dn0)."""
    w = np.asarray(weights, dtype=float)
    T, dT_dt, dT_dn0 = transmittance_and_grads(
        lam_um, t_um, n0, shape, n_inc=n_inc, n_sub=n_sub)
    J = float(T @ w + const)
    return J, dT_dt @ w, dT_dn0 @ w
