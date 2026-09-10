"""Adaptation anchors: the whole loop on a platform the paper never
touched -- a classic two-material dielectric mirror with per-layer
dispersion -- held to an independent textbook closed form and to the
same finite-difference and cross-implementation checks as the native
platform. This is the test that the package generalizes, run rather
than claimed."""
import dataclasses

import numpy as np
import pytest

import fabtwin as ft


def test_quarter_wave_mirror_matches_the_textbook_closed_form():
    """(HL)^p quarter-wave stack at lam0: each quarter-wave layer
    transforms the admittance as Y -> n^2/Y, so the stack presents
    Y = (nH/nL)^(2p) n_sub and R = ((n0 - Y)/(n0 + Y))^2 (Macleod,
    Thin-Film Optical Filters, 4th ed.) -- an independent closed form
    the solver must reproduce exactly, for several p."""
    lam0 = 0.550
    nH, nL, ns = 2.1, 1.5, 1.46
    for p in (1, 3, 6):
        n0 = np.array([nH, nL] * p)
        t = lam0 / (4.0 * n0)                    # quarter-wave each
        R, T = ft.stack_rt(np.array([lam0]), t,
                           n0[:, None] * np.ones((1, 1)), 1.0,
                           np.array([ns]))
        Y = (nH / nL) ** (2 * p) * ns
        R_closed = ((1.0 - Y) / (1.0 + Y)) ** 2
        assert abs(R[0] - R_closed) < 1e-12
        assert abs(R[0] + T[0] - 1.0) < 1e-12


def _two_material_platform(L=81):
    """A 10-layer alternating Si3N4/SiO2 stack on fused silica, both
    materials cited built-ins, with true per-layer dispersion."""
    lam = np.linspace(0.40, 0.80, L)
    lam0 = 0.550
    S_H = ft.dispersion_shape(lam, lam0, ft.SI3N4_LUKE2015)
    S_L = ft.dispersion_shape(lam, lam0, ft.SIO2_MALITSON1965)
    S = np.stack([S_H if i % 2 == 0 else S_L for i in range(10)])
    nsub = ft.SIO2_MALITSON1965.n(lam)
    return lam, lam0, S, nsub


def test_per_layer_shape_reduces_to_shared_shape_exactly():
    lam, lam0, S, nsub = _two_material_platform()
    rng = np.random.default_rng(0)
    t = rng.uniform(0.03, 0.11, 10)
    n0 = rng.uniform(1.5, 2.3, 10)
    shared = S[0]
    tiled = np.tile(shared, (10, 1))
    T1, g1, h1 = ft.transmittance_and_grads(lam, t, n0, shared,
                                            n_sub=nsub)
    T2, g2, h2 = ft.transmittance_and_grads(lam, t, n0, tiled,
                                            n_sub=nsub)
    assert np.abs(T1 - T2).max() == 0.0
    assert np.abs(g1 - g2).max() == 0.0
    assert np.abs(h1 - h2).max() == 0.0


def test_per_layer_adjoint_matches_finite_differences():
    lam, lam0, S, nsub = _two_material_platform()
    rng = np.random.default_rng(1)
    t = rng.uniform(0.03, 0.11, 10)
    n0 = rng.uniform(1.5, 2.3, 10)
    w, c0 = ft.notch_weights(lam, 0.532, 0.015, 0.030)
    _, gd, gn = ft.merit_and_grad(lam, t, n0, S, w, c0, n_sub=nsub)
    h = 1e-7

    def J(tt, nn):
        T = ft.transmittance(lam, tt, nn[:, None] * S, 1.0, nsub)
        return float(ft.merit(T, w, c0))

    rel = []
    for i in range(10):
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
    assert np.median(rel) < 1e-7
    assert np.max(rel) < 1e-5


def test_per_layer_jax_solver_agrees_when_available():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from fabtwin import twin_jax as tj
    lam, lam0, S, nsub = _two_material_platform()
    rng = np.random.default_rng(2)
    t = rng.uniform(0.03, 0.11, 10)
    n0 = rng.uniform(1.5, 2.3, 10)
    Tj = np.asarray(tj.transmittance_jax(lam, jnp.asarray(t),
                                         jnp.asarray(n0), S,
                                         n_sub=jnp.asarray(nsub)))
    Tn = ft.transmittance(lam, t, n0[:, None] * S, 1.0, nsub)
    assert np.abs(Tj - Tn).max() < 1e-12


def test_shape_dimension_mismatches_are_refused():
    lam, lam0, S, nsub = _two_material_platform()
    with pytest.raises(ValueError):
        ft.transmittance_and_grads(lam, np.full(10, 0.05),
                                   np.full(10, 2.0), S[:, :5],
                                   n_sub=nsub)
    with pytest.raises(ValueError):
        ft.transmittance_and_grads(lam, np.full(10, 0.05),
                                   np.full(10, 2.0), S[0][:5],
                                   n_sub=nsub)


def test_user_registered_material_works_end_to_end():
    """A user's own SellmeierMaterial (synthetic test coefficients,
    labeled as such) drops into the same pipeline: range refusal,
    shape construction, solver, adjoint-vs-FD."""
    mat = ft.SellmeierMaterial(name="test material (synthetic)",
                               terms=((1.5, 0.10),), lam_min=0.3,
                               lam_max=2.0, reference="test fixture")
    lam = np.linspace(0.45, 0.65, 21)
    with pytest.raises(ValueError):
        mat.n(np.array([0.2]))
    S = ft.dispersion_shape(lam, 0.550, mat)
    assert abs(float(ft.dispersion_shape(np.array([0.550]), 0.550,
                                         mat)[0]) - 1.0) < 1e-15
    t = np.full(4, 0.06)
    n0 = np.full(4, 1.9)
    w = np.full(21, 1.0 / 21)
    _, gd, _ = ft.merit_and_grad(lam, t, n0, S, w, 0.0, n_sub=1.46)
    h = 1e-7

    def J(tt):
        return float(ft.merit(ft.transmittance(
            lam, tt, n0[:, None] * S[None, :], 1.0, 1.46), w, 0.0))

    tp, tm = t.copy(), t.copy()
    tp[1] += h
    tm[1] -= h
    fd = (J(tp) - J(tm)) / (2 * h)
    assert abs(gd[1] - fd) / max(abs(fd), 1e-12) < 1e-6


def test_full_yield_loop_on_the_adapted_platform():
    """The complete FabGAN-ID loop -- design, traces from a re-scaled
    process, twin fit, CVaR robustification, scoring -- on the
    two-material mirror platform with per-layer dispersion. The
    assertions are structural: robustification must not degrade the
    twin-evaluated CVaR it ascends, and every stage must consume the
    per-layer shapes without special-casing."""
    lam, lam0, S, nsub = _two_material_platform(L=41)
    w, c0 = ft.notch_weights(lam, 0.532, 0.015, 0.030)
    box = ft.DesignBox(10, 0.020, 0.120, 1.5, 2.3)

    t, n0, J, _ = ft.inverse_design(lam, S, w, c0, box, n_probe=40,
                                    n_seed=2, n_iter=20, n_sub=nsub,
                                    seed=3)
    assert J > 0.5                     # better than the trivial J = 1/2

    proc = dataclasses.replace(ft.PAPER_PROCESS, sig_t=0.03,
                               rho=0.4, p_flake=0.02)
    rng = np.random.default_rng(4)
    recipes = box.sample(rng, 40)
    rt, rn, ftd, fnd = proc.trace_dataset(*recipes, 2, rng)
    twin = ft.GaussianTwin(ft.errors_from_traces(rt, rn, ftd, fnd))

    def twin_cvar(tt, nn, seed):
        r = np.random.default_rng(seed)
        return ft.evaluate_under_process(
            lambda a, b, K: twin.sample(r, a, b, K), tt, nn, lam, S,
            w, c0, K=300, n_sub=nsub, alpha=0.1)["CVaR"]

    before = twin_cvar(t, n0, 7)
    t_r, n_r, _ = ft.robustify(lam, t, n0, S, w, c0, twin, box,
                               alpha=0.1, K=32, steps=25, lr=3e-3,
                               seed=0, n_sub=nsub)
    after = twin_cvar(t_r, n_r, 7)
    assert after >= before - 1e-6
    report = ft.evaluate_under_process(
        lambda a, b, K: proc.ensemble(a, b, K,
                                      np.random.default_rng(9)),
        t_r, n_r, lam, S, w, c0, K=300, n_sub=nsub, alpha=0.1)
    assert np.isfinite(report["CVaR"]) and report["CVaR"] <= report["mean"]
