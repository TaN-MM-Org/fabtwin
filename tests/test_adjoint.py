"""Adjoint anchors: forward consistency with the solver, agreement
with central finite differences at the paper's own accuracy figure,
the exactly vanishing gradient at the closed-form quarter-wave
optimum, and refusal of the absorbing case it deliberately does not
cover."""
import numpy as np
import pytest

import fabtwin as ft

LAM = np.linspace(0.40, 0.80, 161)
S = ft.dispersion_shape(LAM, 0.550)
NSUB = ft.SIO2_MALITSON1965.n(LAM)


def test_forward_pass_equals_the_solver():
    rng = np.random.default_rng(0)
    t = rng.uniform(0.020, 0.120, 20)
    n0 = rng.uniform(1.6, 2.4, 20)
    T, _, _ = ft.transmittance_and_grads(LAM, t, n0, S, n_sub=NSUB)
    T_ref = ft.transmittance(LAM, t, n0[:, None] * S[None, :], 1.0, NSUB)
    assert np.abs(T - T_ref).max() < 1e-13


def test_gradients_match_central_finite_differences():
    """The paper reports a 1e-10 median relative error for its JAX
    adjoint against central differences; the hand adjoint matches the
    same yardstick (the max is finite-difference-limited)."""
    rng = np.random.default_rng(0)
    t = rng.uniform(0.020, 0.120, 20)
    n0 = rng.uniform(1.6, 2.4, 20)
    w, c0 = ft.notch_weights(LAM, 0.532, 0.015, 0.030)
    _, gd, gn = ft.merit_and_grad(LAM, t, n0, S, w, c0, n_sub=NSUB)
    h = 1e-7

    def J(tt, nn):
        T = ft.transmittance(LAM, tt, nn[:, None] * S[None, :], 1.0, NSUB)
        return float(ft.merit(T, w, c0))

    rel = []
    for i in range(20):
        tp, tm = t.copy(), t.copy()
        tp[i] += h
        tm[i] -= h
        fd = (J(tp, n0) - J(tm, n0)) / (2 * h)
        rel.append(abs(gd[i] - fd) / max(abs(fd), 1e-12))
        np_, nm = n0.copy(), n0.copy()
        np_[i] += h
        nm[i] -= h
        fd = (J(t, np_) - J(t, nm)) / (2 * h)
        rel.append(abs(gn[i] - fd) / max(abs(fd), 1e-12))
    assert np.median(rel) < 1e-8
    assert np.max(rel) < 1e-6


def test_gradient_vanishes_at_the_quarter_wave_optimum():
    """T(lam0) = 1 is a global maximum, so both partials are exactly
    zero at the closed-form design -- no finite-difference tolerance
    needed."""
    lam0 = 0.550
    lam = np.array([lam0])
    _, gd, gn = ft.merit_and_grad(lam, np.array([lam0 / 8.0]),
                                  np.array([2.0]), np.ones(1),
                                  np.array([1.0]), 0.0,
                                  n_sub=np.array([4.0]))
    assert abs(gd[0]) < 1e-12
    assert abs(gn[0]) < 1e-12


def test_absorbing_designs_are_refused():
    with pytest.raises(ValueError):
        ft.transmittance_and_grads(LAM, np.full(3, 0.05 + 0.001j),
                                   np.full(3, 2.0), S, n_sub=NSUB)


def test_agrees_with_jax_autodiff_when_available():
    """Two independent derivations of the same discrete adjoint --
    hand algebra here, reverse-mode autodiff in the twin extra --
    must agree to machine precision."""
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from fabtwin import twin_jax as tj
    rng = np.random.default_rng(1)
    t = rng.uniform(0.020, 0.120, 12)
    n0 = rng.uniform(1.6, 2.4, 12)
    w, c0 = ft.notch_weights(LAM, 0.532, 0.015, 0.030)
    g = jax.grad(lambda tt, nn: tj.merit_jax(LAM, tt, nn, S, w, c0,
                                             n_sub=jnp.asarray(NSUB)),
                 argnums=(0, 1))(jnp.asarray(t), jnp.asarray(n0))
    _, gd, gn = ft.merit_and_grad(LAM, t, n0, S, w, c0, n_sub=NSUB)
    assert np.abs(np.asarray(g[0]) - gd).max() < 1e-12
    assert np.abs(np.asarray(g[1]) - gn).max() < 1e-12
