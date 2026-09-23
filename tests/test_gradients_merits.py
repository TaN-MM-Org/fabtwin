"""Exact gradients at any angle and with absorption, nonlinear merits,
and the design path running through them (0.7.0)."""
import numpy as np
import pytest

import fabtwin as ft
from fabtwin.gradients import layer_indices, stack_rta_and_grads
from fabtwin.merits import (FunctionMerit, LinearMerit, OpticalModel,
                            SpecMarginMerit, TargetMerit,
                            model_merit_and_grad, model_spectra)

LAM = np.linspace(0.45, 0.80, 7)


def _fwd(t, n, th, pol, ninc, nsub):
    if pol == "u":
        a = ft.stack_rt(LAM, t, n, ninc, nsub, th, "s")
        b = ft.stack_rt(LAM, t, n, ninc, nsub, th, "p")
        return 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
    return ft.stack_rt(LAM, t, n, ninc, nsub, th, pol)


def test_forward_matches_the_open_tmm_reference_beyond_critical_angle():
    """0.7.0 fix: beyond the critical angle of a lossless medium the
    evanescent root must decay; 0.6.1 took the wrong one, which gave a
    wrong R whenever the stack also absorbed."""
    byrnes = pytest.importorskip("tmm")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        N = rng.integers(1, 9)
        t = rng.uniform(0.02, 0.2, N)
        n = rng.uniform(1.0, 2.6, N) + 1j * rng.uniform(0, 0.3, N) * \
            (rng.random(N) < 0.5)
        ninc = rng.choice([1.0, 1.52, 2.0])
        th = rng.uniform(0, 1.5)
        pol = rng.choice(["s", "p"])
        lam = rng.uniform(0.4, 0.9)
        nsub = rng.choice([1.46, 1.0, 3.5, 1.33, 2.0 + 0.3j])
        res = byrnes.coh_tmm(pol, [ninc] + list(n) + [nsub],
                             [np.inf] + list(t * 1000) + [np.inf], th,
                             lam * 1000)
        R, T = ft.stack_rt(np.array([lam]), t, n[:, None], ninc,
                           np.array([nsub]), th, pol)
        worst = max(worst, abs(res["R"] - R[0]), abs(res["T"] - T[0]))
    assert worst < 1e-12
    # the case that was wrong: glass -> absorbing layer -> air at 0.8 rad
    n = np.array([1.38, 2.1 + 0.05j])
    t = np.array([0.1, 0.08])
    res = byrnes.coh_tmm("s", [1.52] + list(n) + [1.0],
                         [np.inf] + list(t * 1000) + [np.inf], 0.8, 550)
    R, _ = ft.stack_rt(np.array([0.55]), t, n[:, None], 1.52,
                       np.array([1.0]), 0.8, "s")
    assert abs(R[0] - res["R"]) < 1e-12


def test_gradients_equal_the_normal_incidence_adjoint():
    rng = np.random.default_rng(3)
    t = rng.uniform(0.03, 0.12, 10)
    n0 = rng.uniform(1.6, 2.4, 10)
    S = ft.dispersion_shape(LAM, 0.55)
    T, dt, dn = ft.transmittance_and_grads(LAM, t, n0, S, 1.0, 1.46)
    g = stack_rta_and_grads(LAM, t, n0[:, None] * S, 1.0, 1.46, 0.0, "s")
    assert np.abs(g["T"] - T).max() < 1e-13
    assert np.abs(g["dT_dt"] - dt).max() < 1e-12 * np.abs(dt).max()
    assert np.abs(g["dT_dn"] * S - dn).max() < 1e-12 * np.abs(dn).max()


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_gradients_match_finite_differences(seed):
    rng = np.random.default_rng(seed)
    for _ in range(10):
        N = rng.integers(1, 7)
        t = rng.uniform(0.03, 0.2, N)
        n = (rng.uniform(1.3, 2.5, (N, 1)) * np.ones((1, LAM.size))
             + 1j * rng.uniform(0.01, 0.2, (N, 1))
             * (rng.random((N, 1)) < 0.6))
        ninc = rng.choice([1.0, 1.52])
        th = rng.uniform(0, 1.3)
        pol = rng.choice(["s", "p", "u"])
        nsub = rng.choice([1.46, 1.0, 3.5 + 0.2j])
        g = stack_rta_and_grads(LAM, t, n, ninc, nsub, th, pol)
        R, T = _fwd(t, n, th, pol, ninc, nsub)
        assert np.abs(g["R"] - R).max() < 1e-13
        assert np.abs(g["T"] - T).max() < 1e-13
        h = 1e-6
        for i in range(N):
            for par in ("t", "n", "k"):
                tp, tm, npl, nm = t.copy(), t.copy(), n.copy(), n.copy()
                one_sided = par == "k" and np.all(n[i].imag == 0)
                if par == "t":
                    tp[i] += h
                    tm[i] -= h
                elif par == "n":
                    npl[i] += h
                    nm[i] -= h
                else:
                    npl[i] += 1j * h
                    if not one_sided:
                        nm[i] -= 1j * h
                den = h if one_sided else 2 * h
                tol = 1e-4 if one_sided else 1e-7
                Rp, Tp = _fwd(tp, npl, th, pol, ninc, nsub)
                Rm, Tm = _fwd(tm, nm, th, pol, ninc, nsub)
                for q, fd in (("R", (Rp - Rm) / den), ("T", (Tp - Tm) / den),
                              ("A", ((1 - Rp - Tp) - (1 - Rm - Tm)) / den)):
                    an = g[f"d{q}_d{par}"][i]
                    # relative to the largest derivative, with an
                    # absolute floor for the finite-difference
                    # round-off (1e-16 / 1e-6 ~ 1e-10) where it is 0
                    assert np.abs(fd - an).max() <= \
                        tol * np.abs(an).max() + 1e-8


def test_gradients_match_jax_autodiff():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    jax.config.update("jax_enable_x64", True)
    # an independent JAX implementation (Byrnes-style angles, n + i k,
    # s polarization) differentiated automatically
    lam = jnp.asarray(LAM)
    ninc, th, nsub = 1.0, 0.7, 1.46
    s0 = ninc * np.sin(th)

    def T_of(t, a, k):
        n = a + 1j * k
        kz = jnp.sqrt(n ** 2 - s0 ** 2 + 0j)
        kz = jnp.where(jnp.imag(kz) < 0, -kz, kz)       # decaying root
        kz0 = np.sqrt(ninc ** 2 - s0 ** 2)
        kzs = np.sqrt(nsub ** 2 - s0 ** 2)
        M00 = jnp.ones_like(lam, dtype=complex)
        M01 = jnp.zeros_like(lam, dtype=complex)
        M10 = jnp.zeros_like(lam, dtype=complex)
        M11 = jnp.ones_like(lam, dtype=complex)
        for i in range(t.shape[0]):
            d = 2 * jnp.pi * kz[i] * t[i] / lam
            c, s = jnp.cos(d), jnp.sin(d)
            e = kz[i]
            m00, m01, m10, m11 = c, -1j * s / e, -1j * e * s, c
            M00, M01, M10, M11 = (M00 * m00 + M01 * m10,
                                  M00 * m01 + M01 * m11,
                                  M10 * m00 + M11 * m10,
                                  M10 * m01 + M11 * m11)
        D = kz0 * M00 + M10 + kz0 * kzs * M01 + kzs * M11
        return jnp.sum(4 * kz0 * kzs / jnp.abs(D) ** 2)
    t = jnp.array([0.08, 0.11, 0.05])
    a = jnp.array([2.1, 1.5, 2.3])
    k = jnp.array([0.05, 0.0, 0.1])
    gt, ga, gk = jax.grad(T_of, argnums=(0, 1, 2))(t, a, k)
    n = (np.asarray(a) + 1j * np.asarray(k))[:, None] * np.ones((1, LAM.size))
    g = stack_rta_and_grads(LAM, np.asarray(t), n, ninc, nsub, th, "s")
    assert abs(float(T_of(t, a, k)) - g["T"].sum()) < 1e-12
    assert np.allclose(np.asarray(gt), g["dT_dt"].sum(1), rtol=1e-10,
                       atol=1e-12)
    assert np.allclose(np.asarray(ga), g["dT_dn"].sum(1), rtol=1e-10,
                       atol=1e-12)
    assert np.allclose(np.asarray(gk), g["dT_dk"].sum(1), rtol=1e-10,
                       atol=1e-12)


def _model_fd(model, t, n0, S, ninc, nsub):
    J, dt, dn = model_merit_and_grad(model, LAM, t, n0, S, ninc, nsub)
    h = 1e-6
    for i in range(t.size):
        for arr, g in ((t, dt), (n0, dn)):
            ap, am = arr.copy(), arr.copy()
            ap[i] += h
            am[i] -= h
            tt = (ap, n0) if arr is t else (t, ap)
            tm = (am, n0) if arr is t else (t, am)
            Jp = model_merit_and_grad(model, LAM, *tt, S, ninc, nsub)[0]
            Jm = model_merit_and_grad(model, LAM, *tm, S, ninc, nsub)[0]
            fd = (Jp - Jm) / (2 * h)
            assert abs(fd - g[i]) <= 1e-6 * (abs(g[i]) + 1e-6)


def test_merit_gradients_match_finite_differences():
    rng = np.random.default_rng(5)
    S = ft.dispersion_shape(LAM, 0.55)
    t = rng.uniform(0.05, 0.12, 5)
    n0 = rng.uniform(1.7, 2.3, 5)
    K = np.full(LAM.size, 0.02)
    conds = ((0.0, "s"), (0.6, "u"), (1.0, "p"))
    stop = LAM < 0.55
    merits = [
        LinearMerit(rng.normal(size=LAM.size), 0.3, "R"),
        LinearMerit(rng.normal(size=(3, LAM.size)), 0.0, "A"),
        TargetMerit(np.linspace(0.2, 0.9, LAM.size), "T"),
        SpecMarginMerit(stop, ~stop, leak_max=0.5, pass_min=0.4,
                        sharpness=40.0),
        FunctionMerit(lambda R, T, A: (np.sum(T ** 2), 0 * R, 2 * T,
                                       0 * A)),
    ]
    for m in merits:
        _model_fd(OpticalModel(m, conds, kext=K), t, n0, S, 1.0, 1.46)


def test_spec_margin_positive_implies_pass():
    rng = np.random.default_rng(6)
    L = 40
    stop = np.zeros(L, bool)
    stop[15:22] = True
    pas = np.zeros(L, bool)
    pas[:10] = True
    pas[28:] = True
    m = SpecMarginMerit(stop, pas, leak_max=0.1, pass_min=0.9,
                        sharpness=100.0)
    n_pos = 0
    for _ in range(3000):
        T = np.clip(rng.normal(0.9, 0.08, L), 0, 1)
        T[stop] = np.clip(rng.normal(0.05, 0.04, stop.sum()), 0, 1)
        J = m(0 * T[None], T[None], 0 * T[None])[0]
        ok = ft.pass_fail(T, stop, pas, 0.1, 0.9)
        m1, m2 = m.margins(T)
        assert J <= min(m1[0], m2[0]) + 1e-12       # never optimistic
        if J > 0:
            n_pos += 1
            assert ok
    assert n_pos > 100


def test_linear_merit_model_equals_the_original_path():
    rng = np.random.default_rng(7)
    S = ft.dispersion_shape(LAM, 0.55)
    t = rng.uniform(0.05, 0.12, 6)
    n0 = rng.uniform(1.7, 2.3, 6)
    w = rng.normal(size=LAM.size)
    J0, dt0, dn0 = ft.merit_and_grad(LAM, t, n0, S, w, 0.2, 1.0, 1.46)
    model = OpticalModel(LinearMerit(w, 0.2, "T"))
    J1, dt1, dn1 = model_merit_and_grad(model, LAM, t, n0, S, 1.0, 1.46)
    assert abs(J0 - J1) < 1e-13
    assert np.abs(dt0 - dt1).max() < 1e-11 * np.abs(dt0).max()
    assert np.abs(dn0 - dn1).max() < 1e-11 * np.abs(dn0).max()


def test_design_at_an_angle_with_absorption():
    # a 45-degree, unpolarized, absorbing bandpass: the model path
    # designs it and never loses to random search with the same number
    # of merit evaluations
    lam = np.linspace(0.45, 0.70, 51)
    S = np.ones(lam.size)
    w, c = ft.bandpass_weights(lam, 0.55, 0.60, 0.02)
    model = OpticalModel(LinearMerit(w, c, "T"), ((np.pi / 4, "u"),),
                         kext=np.full(lam.size, 0.003))
    box = ft.DesignBox(6, 0.03, 0.20, 1.45, 2.3)
    t, n, J, traj = ft.inverse_design(lam, S, None, None, box,
                                      n_probe=60, n_seed=2, n_iter=30,
                                      n_sub=1.46, model=model)
    _, _, Jr, _ = ft.random_search(lam, S, None, None, box, 60 + 60,
                                   n_sub=1.46, model=model)
    assert J >= Jr
    R, T, A = model_spectra(model, lam, t, n, S, 1.0, 1.46)
    assert np.all(A > 0)                   # it does absorb
    assert abs(J - (T[0] @ w + c)) < 1e-12


def test_refusals():
    with pytest.raises(ValueError):
        stack_rta_and_grads(LAM, np.array([0.1]), np.array([2.0]),
                            1.0 + 0.1j, 1.46)
    with pytest.raises(ValueError):
        stack_rta_and_grads(LAM, np.array([0.1]), np.array([2.0 - 0.1j]))
    with pytest.raises(ValueError):
        stack_rta_and_grads(LAM, np.array([0.1]), np.array([2.0]),
                            pol="x")
    with pytest.raises(ValueError):
        stack_rta_and_grads(LAM, np.array([0.1]), np.array([2.0]),
                            theta0_rad=np.pi / 2)
    with pytest.raises(ValueError):
        OpticalModel(LinearMerit(np.ones(3)), ((2.0, "s"),))
    with pytest.raises(ValueError):
        layer_indices([2.0], np.ones(3), kext=-np.ones(3))
    with pytest.raises(ValueError):
        ft.inverse_design(LAM, np.ones(LAM.size), None, None,
                          ft.DesignBox(2, 0.05, 0.1, 1.5, 2.0))
    with pytest.raises(ValueError):
        SpecMarginMerit(np.zeros(5, bool), np.ones(5, bool))


def test_scoring_functions_accept_a_model():
    rng = np.random.default_rng(8)
    lam = np.linspace(0.45, 0.70, 21)
    S = np.ones(lam.size)
    w, c = ft.bandpass_weights(lam, 0.55, 0.60, 0.02)
    t = rng.uniform(0.05, 0.12, 5)
    n0 = rng.uniform(1.6, 2.2, 5)
    model = OpticalModel(LinearMerit(w, c, "T"))       # normal incidence
    x = rng.normal(0, 0.01, (20, 10))
    a = ft.induced_merits(x, t, n0, lam, S, w, c, 1.0, 1.46)
    b = ft.induced_merits(x, t, n0, lam, S, None, None, 1.0, 1.46,
                          model=model)
    assert np.allclose(a, b, atol=1e-13)
    samp = lambda tt, nn, K: ft.apply_errors(tt, nn, x[:K])
    ea = ft.evaluate_under_process(samp, t, n0, lam, S, w, c, 20,
                                   n_sub=1.46)
    eb = ft.evaluate_under_process(samp, t, n0, lam, S, None, None, 20,
                                   n_sub=1.46, model=model)
    assert np.allclose(ea["samples"], eb["samples"], atol=1e-13)


def test_audit_fixes_and_refusals():
    # stack_rt accepts unpolarized light as the mean of s and p
    t = np.array([0.08, 0.1])
    n = np.array([[2.0], [1.5 + 0.02j]]) * np.ones((1, LAM.size))
    Ru, Tu = ft.stack_rt(LAM, t, n, 1.0, 1.46, 0.7, "u")
    Rs, Ts = ft.stack_rt(LAM, t, n, 1.0, 1.46, 0.7, "s")
    Rp, Tp = ft.stack_rt(LAM, t, n, 1.0, 1.46, 0.7, "p")
    assert np.allclose(Ru, (Rs + Rp) / 2, atol=1e-15)
    assert np.allclose(Tu, (Ts + Tp) / 2, atol=1e-15)
    # a substrate exactly at its critical angle has no finite p
    # admittance: refused instead of NaN
    th_c = np.arcsin(1.0 / 1.52)
    with pytest.raises(ValueError, match="critical"):
        ft.stack_rt(LAM, t, n, 1.52, 1.0, th_c, "p")
    with pytest.raises(ValueError, match="critical"):
        stack_rta_and_grads(LAM, t, n, 1.52, 1.0, th_c, "p")
    # a 1-D kext is k(lam); a wrong length is refused with a message
    with pytest.raises(ValueError, match="per wavelength"):
        layer_indices([2.0, 1.5], np.ones(LAM.size), kext=[0.1, 0.2])
    assert layer_indices([2.0, 1.5], np.ones(LAM.size),
                         kext=[[0.1], [0.2]]).shape == (2, LAM.size)
    # weights and a model together are refused, not silently mixed
    box = ft.DesignBox(2, 0.05, 0.1, 1.5, 2.0)
    model = OpticalModel(LinearMerit(np.ones(LAM.size)), (0.0, "s"))
    assert model.conditions == ((0.0, "s"),)       # a single pair works
    with pytest.raises(ValueError, match="not both"):
        ft.inverse_design(LAM, np.ones(LAM.size), np.ones(LAM.size), 0.0,
                          box, model=model)
    # a mask of the wrong length is a ValueError, not an IndexError
    m = SpecMarginMerit(np.r_[True, False], np.r_[False, True])
    with pytest.raises(ValueError, match="mask"):
        m(np.zeros((1, 5)), np.zeros((1, 5)), np.zeros((1, 5)))
    # tuple group labels work in mondrian_quantiles
    q = ft.mondrian_quantiles(np.arange(40.0), [("a", 1)] * 20
                              + [("b", 2)] * 20, 0.1)
    assert set(q) == {("a", 1), ("b", 2)}


def test_joint_recovery_input_errors_are_named():
    lam = np.linspace(0.45, 0.8, 30)
    S = np.ones(lam.size)
    g = stack_rta_and_grads(lam, np.array([0.1, 0.08]),
                            np.array([2.0, 1.5]), 1.0, 1.46, 0.0, "s")
    ok = ft.Measurement(0.0, "s", "T", g["T"], 0.002)
    with pytest.raises(ValueError, match="pol"):
        ft.errors_from_spectra(lam, [ft.Measurement(0.0, "x", "T", g["T"])],
                               [0.1, 0.08], [2.0, 1.5], S, n_sub=1.46)
    with pytest.raises(ValueError, match="theta0_rad"):
        ft.errors_from_spectra(lam, [ft.Measurement(2.0, "s", "T", g["T"],
                                                    0.002)],
                               [0.1, 0.08], [2.0, 1.5], S, n_sub=1.46)
    with pytest.raises(ValueError, match="sigma"):
        ft.errors_from_spectra(lam, [ok, ft.Measurement(0.0, "p", "T",
                                                        g["T"])],
                               [0.1, 0.08], [2.0, 1.5], S, n_sub=1.46)
