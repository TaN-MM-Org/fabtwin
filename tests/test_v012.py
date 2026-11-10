"""Tests of 0.12.0: a thick substrate that absorbs (its thickness given
as `sub_thickness_um`), and four fixes: the transmittance from an
absorbing incidence medium, NaN merits in `cvar`, the variance of the
diagonal Gaussian twin, and complex inputs to the JAX solver.

Independent references: the incoherent solver `inc_tmm` and the
coherent solver `coh_tmm` of the `tmm` package (S. J. Byrnes,
arXiv:1603.02720); the closed form of a bare absorbing plate, written
out here by hand; central finite differences of the forward optics;
the textbook unbiased variance (a - b)^2 / 2 of two values; and plain
arithmetic for the tail of a short list.
"""
import warnings

import numpy as np
import pytest

import fabtwin as ft
from fabtwin.gradients import stack_rta_and_grads
from fabtwin.merits import (LinearMerit, OpticalModel, TargetMerit,
                            model_merit_and_grad, model_spectra)


def _inc_tmm(byrnes, pol, lam, t, n, ninc, nsub, nexit, th, d_um):
    """`tmm.inc_tmm`: coherent layers on an incoherent substrate of
    thickness d_um (um), which may absorb, between incoherent media."""
    def one(p):
        r = byrnes.inc_tmm(p, [ninc] + list(n) + [nsub, nexit],
                           [np.inf] + list(np.asarray(t) * 1000)
                           + [d_um * 1000, np.inf],
                           ["i"] + ["c"] * len(t) + ["i", "i"], th,
                           lam * 1000)
        return r["R"], r["T"]
    if pol == "u":
        (Rs, Ts), (Rp, Tp) = one("s"), one("p")
        return 0.5 * (Rs + Rp), 0.5 * (Ts + Tp)
    return one(pol)


# --------------------- a thick substrate that absorbs ---------------------


def test_absorbing_substrate_matches_the_open_tmm_incoherent_solver():
    """200 random plates: absorbing and lossless layers, s, p and
    unpolarized light, angles up to 80 degrees, substrate k from 1e-8 to
    1e-3 and thickness 0.1 to 3 mm (single-pass power kept from about
    1e-4 to 1; tmm clamps it at 1e-30, so far more opaque plates are not
    compared), incidence and exit media of index 1 and 1.33."""
    byrnes = pytest.importorskip("tmm")
    rng = np.random.default_rng(12)
    worst, compared = 0.0, 0
    while compared < 200:
        N = int(rng.integers(0, 6))
        t = rng.uniform(0.02, 0.2, N)
        n = rng.uniform(1.3, 2.5, N) + 1j * rng.uniform(0, 0.3, N) * \
            (rng.random(N) < 0.5)
        ninc = float(rng.choice([1.0, 1.33]))
        nexit = float(rng.choice([1.0, 1.33]))
        nsub = rng.uniform(1.4, 1.9) + 1j * 10 ** rng.uniform(-8, -3)
        d = rng.uniform(100.0, 3000.0)
        th = rng.uniform(0, 1.4)
        pol = str(rng.choice(["s", "p", "u"]))
        lam = rng.uniform(0.4, 0.9)
        if 4 * np.pi * d * nsub.imag / lam > 9:      # tau below ~1e-4
            continue
        R, T = ft.stack_rt(np.array([lam]), t, n[:, None], ninc, nsub, th,
                           pol, n_exit=nexit, sub_thickness_um=d)
        Rb, Tb = _inc_tmm(byrnes, pol, lam, t, n, ninc, nsub, nexit, th, d)
        worst = max(worst, abs(R[0] - Rb), abs(T[0] - Tb))
        compared += 1
    assert worst < 1e-12
    # a dispersive absorbing substrate given as an (L,) array, one call
    lam = np.linspace(0.4, 0.8, 7)
    nsub = ft.SIO2_MALITSON1965.n(lam) + 1j * np.linspace(2e-6, 5e-7, 7)
    t = np.array([0.09, 0.07, 0.012])
    n = np.array([2.05, 1.46, 0.5 + 2.5j])
    R, T = ft.stack_rt(lam, t, n, 1.0, nsub, 0.5, "u", n_exit=1.0,
                       sub_thickness_um=1000.0)
    for j in range(lam.size):
        Rb, Tb = _inc_tmm(byrnes, "u", lam[j], t, n, 1.0, nsub[j], 1.0, 0.5,
                          1000.0)
        assert abs(R[j] - Rb) < 1e-12 and abs(T[j] - Tb) < 1e-12


def test_bare_absorbing_plate_equals_its_closed_form():
    """A bare plate of index n = a + i k and thickness d in air, at
    normal incidence. With r = (n - 1) / (n + 1), R1 = |r|^2, the power
    entering the plate Tf = 4 a / |1 + n|^2, the power leaving it per
    unit power arriving at a face from inside Tb = 4 |n|^2 / (a |1 + n|^2)
    and tau = exp(-4 pi k d / lam):

        T = Tf Tb tau / (1 - R1^2 tau^2),
        R = R1 + Tf Tb R1 tau^2 / (1 - R1^2 tau^2).

    Also: for k -> 0 the result goes to the lossless plate 2a / (a^2 + 1),
    and a lossless substrate gives the same numbers with or without a
    thickness (the thickness is not used then)."""
    lam = np.array([0.45, 0.6, 0.8])
    for a, k, d in ((1.5, 1e-6, 1000.0), (1.52, 3e-5, 500.0),
                    (1.9, 2e-4, 2000.0), (1.46, 1e-3, 100.0)):
        n = a + 1j * k
        R1 = abs((n - 1) / (n + 1)) ** 2
        Tf = 4 * a / abs(1 + n) ** 2
        Tb = 4 * abs(n) ** 2 / (a * abs(1 + n) ** 2)
        tau = np.exp(-4 * np.pi * k * d / lam)
        T_cf = Tf * Tb * tau / (1 - R1 ** 2 * tau ** 2)
        R_cf = R1 + Tf * Tb * R1 * tau ** 2 / (1 - R1 ** 2 * tau ** 2)
        for pol in ("s", "p", "u"):
            R, T = ft.stack_rt(lam, [], np.zeros((0, 3)), 1.0, n, 0.0, pol,
                               n_exit=1.0, sub_thickness_um=d)
            np.testing.assert_allclose(T, T_cf, rtol=0, atol=1e-14)
            np.testing.assert_allclose(R, R_cf, rtol=0, atol=1e-14)
            g = stack_rta_and_grads(lam, [], np.zeros((0, 3)), 1.0, n, 0.0,
                                    pol, n_exit=1.0, sub_thickness_um=d)
            np.testing.assert_allclose(g["T"], T_cf, rtol=0, atol=1e-14)
            np.testing.assert_allclose(g["A"], 1 - R_cf - T_cf, rtol=0,
                                       atol=1e-14)
            assert np.all(g["A"] > 0)
    # k -> 0: the lossless plate (the absorption is first order in k)
    R, T = ft.stack_rt(lam, [], np.zeros((0, 3)), 1.0, 1.5 + 1e-13j,
                       n_exit=1.0, sub_thickness_um=1000.0)
    np.testing.assert_allclose(T, 2 * 1.5 / (1.5 ** 2 + 1), rtol=0,
                               atol=1e-8)
    # a lossless substrate: the thickness changes nothing (same arithmetic)
    t, nl = np.array([0.08, 0.11]), np.array([2.1, 1.46 + 0.01j])
    for pol in ("s", "p", "u"):
        a = ft.stack_rt(lam, t, nl, 1.0, 1.52, 0.3, pol, n_exit=1.0)
        b = ft.stack_rt(lam, t, nl, 1.0, 1.52, 0.3, pol, n_exit=1.0,
                        sub_thickness_um=1000.0)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_absorbing_substrate_gradients_match_finite_differences():
    """Every derivative (R, T, A with respect to thickness, n and k)
    against central finite differences of `stack_rt`, on 12 random
    stacks on an absorbing plate (relative 1e-7 plus absolute 1e-8, as
    for the lossless plate in 0.8.0)."""
    rng = np.random.default_rng(5)
    h = 1e-6
    for _ in range(12):
        N = int(rng.integers(1, 5))
        L = 4
        lam = np.sort(rng.uniform(0.45, 0.85, L))
        t = rng.uniform(0.03, 0.15, N)
        n = rng.uniform(1.4, 2.4, (N, L)) + 1j * rng.uniform(0.0, 0.2,
                                                             (N, L))
        nsub = rng.uniform(1.45, 1.8) + 1j * 10 ** rng.uniform(-6, -4)
        d = rng.uniform(200.0, 2000.0)
        th = rng.uniform(0, 1.2)
        pol = str(rng.choice(["s", "p", "u"]))
        kw = dict(n_exit=1.0, sub_thickness_um=d)
        g = stack_rta_and_grads(lam, t, n, 1.0, nsub, th, pol, **kw)

        def fwd(tt, nn):
            R, T = ft.stack_rt(lam, tt, nn, 1.0, nsub, th, pol, **kw)
            return dict(R=R, T=T, A=1 - R - T)
        for i in range(N):
            e = np.zeros(N)
            e[i] = h
            E = np.zeros((N, L))
            E[i] = h
            for name, (p, m) in (("t", (fwd(t + e, n), fwd(t - e, n))),
                                 ("n", (fwd(t, n + E), fwd(t, n - E))),
                                 ("k", (fwd(t, n + 1j * E),
                                        fwd(t, n - 1j * E)))):
                for q in ("R", "T", "A"):
                    fd = (p[q] - m[q]) / (2 * h)
                    ex = g[f"d{q}_d{name}"][i]
                    assert np.all(np.abs(ex - fd)
                                  <= 1e-7 * np.abs(fd) + 1e-8), (name, q)


def test_absorbing_substrate_in_the_design_model():
    """`OpticalModel(sub_thickness_um=)`: its spectra equal `stack_rt`
    (1e-13), and its merit gradient matches central finite differences
    (relative 1e-7)."""
    lam = np.linspace(0.45, 0.7, 6)
    S = ft.dispersion_shape(lam, 0.55)
    t = np.array([0.08, 0.10, 0.07])
    n0 = np.array([2.1, 1.5, 2.0])
    nsub = ft.SIO2_MALITSON1965.n(lam) + 5e-6j
    target = np.linspace(0.3, 0.9, lam.size)
    model = OpticalModel(TargetMerit(target, "T"),
                         ((0.0, "s"), (0.4, "u")), n_exit=1.0,
                         sub_thickness_um=1500.0)
    R, T, A = model_spectra(model, lam, t, n0, S, 1.0, nsub)
    for c, (ang, pol) in enumerate(model.conditions):
        Rr, Tr = ft.stack_rt(lam, t, n0[:, None] * S, 1.0, nsub, ang, pol,
                             n_exit=1.0, sub_thickness_um=1500.0)
        np.testing.assert_allclose(R[c], Rr, rtol=0, atol=1e-13)
        np.testing.assert_allclose(T[c], Tr, rtol=0, atol=1e-13)
    J, gt, gn = model_merit_and_grad(model, lam, t, n0, S, 1.0, nsub)
    h = 1e-6
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        fdt = (model_merit_and_grad(model, lam, t + e, n0, S, 1.0, nsub)[0]
               - model_merit_and_grad(model, lam, t - e, n0, S, 1.0,
                                      nsub)[0]) / (2 * h)
        fdn = (model_merit_and_grad(model, lam, t, n0 + e, S, 1.0, nsub)[0]
               - model_merit_and_grad(model, lam, t, n0 - e, S, 1.0,
                                      nsub)[0]) / (2 * h)
        assert abs(gt[i] - fdt) <= 1e-7 * abs(fdt) + 1e-10
        assert abs(gn[i] - fdn) <= 1e-7 * abs(fdn) + 1e-10


def _absorbing_plate_case(byrnes):
    lam = np.linspace(0.40, 0.80, 81)
    S = ft.dispersion_shape(lam, 0.550)
    nsub = ft.SIO2_MALITSON1965.n(lam) + 2e-5j      # a tinted glass
    t0 = np.array([0.070, 0.095, 0.060, 0.110, 0.080])
    n0 = np.array([2.2, 1.5, 2.2, 1.5, 2.2])
    xt = np.array([0.03, -0.02, 0.015, 0.01, -0.025])
    nl = n0[:, None] * S
    T = np.array([_inc_tmm(byrnes, "s", lam[j], t0 * (1 + xt), nl[:, j],
                           1.0, nsub[j], 1.0, 0.0, 1000.0)[1]
                  for j in range(lam.size)])
    return lam, S, nsub, t0, n0, xt, T


def test_recovery_from_an_absorbing_plate():
    """A plate spectrum computed by `tmm.inc_tmm` (1 mm of a glass with
    k = 2e-5, so about 40 % of the light is absorbed in one pass at
    0.6 um): `errors_from_spectrum(n_exit=1.0, sub_thickness_um=1000.0)`
    recovers the thickness errors to 1e-6, `errors_from_spectra` agrees
    with it (1e-8), and `recovery_precision` with the same substrate
    equals the uncertainties the fit reports (1e-6). With 0.2 % noise
    and the absorption left out (the real part of the substrate index),
    the fit is flagged by the chi-square check."""
    byrnes = pytest.importorskip("tmm")
    lam, S, nsub, t0, n0, xt, T = _absorbing_plate_case(byrnes)
    kw = dict(n_sub=nsub, n_exit=1.0, sub_thickness_um=1000.0)
    rec = ft.errors_from_spectrum(lam, T, t0, n0, S, n_starts=4, **kw)
    assert np.abs(rec.dt_over_t - xt).max() < 1e-6
    m = ft.Measurement(0.0, "s", "T", T, None)
    joint = ft.errors_from_spectra(lam, [m], t0, n0, S, n_starts=4, **kw)
    assert np.abs(joint.dt_over_t - rec.dt_over_t).max() < 1e-8

    rng = np.random.default_rng(3)
    Tn = T + rng.normal(0, 2e-3, lam.size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        good = ft.errors_from_spectrum(lam, Tn, t0, n0, S, sigma_T=2e-3,
                                       n_starts=4, **kw)
    assert good.chi2_pvalue > 1e-3
    assert np.all(np.abs(good.dt_over_t - xt) < 4 * good.sigma)
    plan = ft.recovery_precision(
        lam, [ft.Measurement(0.0, "s", "T", None, 2e-3)], t0, n0, S,
        at_dt_over_t=good.dt_over_t, **kw)
    np.testing.assert_allclose(plan.sigma_t, good.sigma, rtol=1e-6)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bad = ft.errors_from_spectrum(lam, Tn, t0, n0, S, sigma_T=2e-3,
                                      n_sub=np.real(nsub), n_exit=1.0,
                                      n_starts=4)
    assert bad.chi2_pvalue < 1e-6
    assert any("chi2" in str(w.message) for w in caught)


def test_absorbing_substrate_refusals():
    lam = np.array([0.5, 0.6])
    t, n = np.array([0.1]), np.array([2.0])
    with pytest.raises(ValueError, match="sub_thickness_um"):
        ft.stack_rt(lam, t, n, 1.0, 1.5 + 1e-5j, n_exit=1.0)
    with pytest.raises(ValueError, match="needs n_exit"):
        ft.stack_rt(lam, t, n, 1.0, 1.5, sub_thickness_um=1000.0)
    with pytest.raises(ValueError, match="needs n_exit"):
        stack_rta_and_grads(lam, t, n, 1.0, 1.5, sub_thickness_um=1000.0)
    for bad in (0.0, -5.0, np.inf, np.nan, "1 mm"):
        with pytest.raises(ValueError, match="sub_thickness_um"):
            ft.stack_rt(lam, t, n, 1.0, 1.5 + 1e-5j, n_exit=1.0,
                        sub_thickness_um=bad)
    with pytest.raises(ValueError, match="gain"):
        ft.stack_rt(lam, t, n, 1.0, 1.5 - 1e-5j, n_exit=1.0,
                    sub_thickness_um=1000.0)
    with pytest.raises(ValueError, match="lossless"):        # exit medium
        ft.stack_rt(lam, t, n, 1.0, 1.5 + 1e-5j, n_exit=1.0 + 0.1j,
                    sub_thickness_um=1000.0)
    with pytest.raises(ValueError, match="plays no role"):
        ft.stack_rt(lam, t, n, 1.8, 1.5 + 1e-5j, 1.2, n_exit=1.0,
                    sub_thickness_um=1000.0)
    with pytest.raises(ValueError, match="needs n_exit"):
        OpticalModel(LinearMerit(np.ones(2)), sub_thickness_um=1000.0)
    with pytest.raises(ValueError, match="positive"):
        OpticalModel(LinearMerit(np.ones(2)), n_exit=1.0,
                     sub_thickness_um=-1.0)
    with pytest.raises(ValueError, match="needs n_exit"):
        ft.errors_from_spectrum(lam, np.array([0.5, 0.5]), t, n,
                                np.ones(2), sub_thickness_um=1000.0)


# ------------------ fix: an absorbing incidence medium ------------------


def test_absorbing_incidence_medium_matches_the_open_tmm_solver():
    """Regression (0.11.0 used the lossless formula
    T = 4 Re(eta0) Re(eta_sub) / |D|^2 for an absorbing incidence medium
    too, so T was low by the factor Re(eta0)^2 / |eta0|^2; for 80 nm of
    index 2.1 on 1.46 at 0.55 um, seen from a medium of 1.5 + 0.3i, it
    gave 0.875537 where `tmm.coh_tmm` gives 0.910558). At normal
    incidence, 100 random stacks, s and p, against `tmm.coh_tmm` to
    1e-12; oblique incidence from such a medium is refused (n sin(theta)
    would be complex; `tmm` refuses it too)."""
    byrnes = pytest.importorskip("tmm")
    R, T = ft.stack_rt(np.array([0.55]), [0.08], np.array([[2.1]]),
                       1.5 + 0.3j, 1.46)
    ref = byrnes.coh_tmm("s", [1.5 + 0.3j, 2.1, 1.46], [np.inf, 80, np.inf],
                         0.0, 550.0)
    assert abs(T[0] - ref["T"]) < 1e-12 and abs(R[0] - ref["R"]) < 1e-12
    assert abs(T[0] - 0.910558) < 1e-6
    rng = np.random.default_rng(8)
    worst = 0.0
    for _ in range(100):
        N = int(rng.integers(0, 6))
        t = rng.uniform(0.02, 0.2, N)
        n = rng.uniform(1.3, 2.5, N) + 1j * rng.uniform(0, 0.3, N) * \
            (rng.random(N) < 0.5)
        ninc = rng.uniform(1.0, 2.0) + 1j * rng.uniform(0.001, 0.5)
        nsub = rng.uniform(1.3, 2.0) + 1j * rng.uniform(0, 0.2) * \
            (rng.random() < 0.5)
        lam = rng.uniform(0.4, 0.9)
        pol = str(rng.choice(["s", "p"]))
        R, T = ft.stack_rt(np.array([lam]), t, n[:, None], ninc, nsub, 0.0,
                           pol)
        ref = byrnes.coh_tmm(pol, [ninc] + list(n) + [nsub],
                             [np.inf] + list(t * 1000) + [np.inf], 0.0,
                             lam * 1000)
        worst = max(worst, abs(R[0] - ref["R"]), abs(T[0] - ref["T"]))
    assert worst < 1e-12
    with pytest.raises(ValueError, match="absorbing incidence medium"):
        ft.stack_rt(np.array([0.55]), [0.08], np.array([[2.1]]),
                    1.5 + 0.3j, 1.46, 0.3, "s")
    # a lossless incidence medium written as a complex number is not
    # absorbing: unchanged, and allowed at an angle
    a = ft.stack_rt(np.array([0.55]), [0.08], np.array([[2.1]]),
                    1.5 + 0.0j, 1.46, 0.3, "p")
    b = ft.stack_rt(np.array([0.55]), [0.08], np.array([[2.1]]), 1.5, 1.46,
                    0.3, "p")
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


# ------------------------- fix: NaN merits in cvar -------------------------


def test_cvar_refuses_nan_merits():
    """Regression: NaN sorts last, so 0.11.0 left a NaN merit out of the
    tail as if it were the best device. cvar([nan, 1, ..., 9], 0.2) gave
    1.5 (the mean of 1 and 2), and tail_statistics reported that CVaR
    next to a NaN mean. Now refused; finite values are unchanged (the
    mean of the 2 smallest of 0..9 is 0.5)."""
    v = np.arange(10.0)
    assert ft.cvar(v, 0.2) == 0.5
    v[0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        ft.cvar(v, 0.2)
    with pytest.raises(ValueError, match="NaN"):
        ft.tail_statistics(v, 0.2)
    with pytest.raises(ValueError, match="NaN"):
        ft.distribution_distances(v, np.arange(10.0), 0.2)
    assert ft.cvar(np.array([-np.inf, 1.0, 2.0]), 0.3) == -np.inf


# ------------------ fix: the variance of the diagonal twin ------------------


def test_diagonal_twin_has_the_variances_of_the_full_twin():
    """Regression: 0.11.0 fitted the diagonal twin with the divisor M
    (numpy's var) and the full twin with M - 1 (numpy's cov), so for 2
    runs the diagonal variances were half the full ones. Reference: the
    unbiased variance of two values a, b is (a - b)^2 / 2."""
    x = np.array([[0.010, -0.004, 0.02],
                  [0.030, 0.002, -0.01]])
    diag = ft.GaussianTwin(x, diagonal=True, jitter=0.0)
    full = ft.GaussianTwin(x)          # rank 1 without its jitter of 1e-9
    ref = (x[0] - x[1]) ** 2 / 2
    np.testing.assert_allclose(np.diag(diag.cov), ref, rtol=1e-12, atol=0)
    np.testing.assert_allclose(np.diag(full.cov) - 1e-9, ref, rtol=1e-9,
                               atol=0)
    assert np.all(diag.cov[~np.eye(3, dtype=bool)] == 0.0)
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.02, (40, 6))
    np.testing.assert_allclose(
        np.diag(ft.GaussianTwin(x, diagonal=True).cov),
        np.diag(ft.GaussianTwin(x).cov), rtol=1e-12, atol=0)
    ens = ft.TwinEnsemble(x, n_members=3, diagonal=True)
    for m in ens.members:
        assert np.all(m.cov[~np.eye(6, dtype=bool)] == 0.0)


# ------------------- fix: complex inputs to the JAX solver -------------------


def test_jax_solver_refuses_complex_inputs():
    """Regression: `transmittance_jax` uses the indices as given, without
    the conjugation `fabtwin.tmm` applies to n + i k, so 0.11.0 returned
    a wrong T for an absorbing substrate (three layers of index 2.0,
    0.06 um, on 1.5 + 0.2i: 0.8249 at 0.45 um, where `fabtwin.tmm` gives
    0.7867) and T above 1 for absorbing layers. Now refused, also inside
    `robustify_gan`; real inputs still equal `fabtwin.tmm` (1e-12)."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from fabtwin import twin_jax as tj
    lam = np.array([0.45, 0.55, 0.65])
    t = jnp.asarray([0.06, 0.06, 0.06])
    n0 = jnp.asarray([2.0, 2.0, 2.0])
    S = np.ones(3)
    T = np.asarray(tj.transmittance_jax(lam, t, n0, S, 1.0, 1.5))
    np.testing.assert_allclose(
        T, ft.transmittance(lam, np.asarray(t), np.full((3, 3), 2.0), 1.0,
                            1.5), rtol=0, atol=1e-12)
    for kw in (dict(n_sub=jnp.full(3, 1.5 + 0.2j)), dict(n_sub=1.5 + 0.2j),
               dict(n_inc=1.0 + 0.1j)):
        with pytest.raises(ValueError, match="lossless"):
            tj.transmittance_jax(lam, t, n0, S, **kw)
    with pytest.raises(ValueError, match="lossless"):
        tj.transmittance_jax(lam, t, n0 + 0.05j, S, 1.0, 1.5)
    with pytest.raises(ValueError, match="lossless"):
        tj.merit_jax(lam, t, n0, S, np.ones(3) / 3, 0.0,
                     n_sub=jnp.full(3, 1.5 + 0.2j))
    cfg = tj.FabTwinConfig(3, 0.03, 0.1, 1.8, 2.2, dim_z=4, hidden=(8,))
    import jax
    gp, _ = tj.init_models(cfg, jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="lossless"):
        tj.robustify_gan(cfg, gp, lam, np.asarray(t), np.asarray(n0), S,
                         np.ones(3) / 3, 0.0, np.full(3, 1.5 + 0.2j), K=8,
                         steps=1)


# --------------- a drifting monitor: one signal offset per stop ---------------

T4 = np.array([0.075, 0.098, 0.066, 0.118])     # README example 22
N4 = np.array([2.1, 1.46, 2.1, 1.46])


def _brentq_run(lam, t, n, e, ns, off):
    """One run simulated independently: layer by layer (layer N - 1
    first), scipy's brentq on `stack_rt` in a bracket of +/- 20 % around
    the recipe (no turning point inside for this recipe at 0.47 um),
    with the offset off[j] at the stop of layer j."""
    from scipy.optimize import brentq
    N = t.size
    dep = np.empty(N)
    for j in range(N - 1, -1, -1):
        cut = ft.stack_rt([lam], t[j:], n[j:], 1.0, ns)[1][0]

        def f(tau):
            return ft.stack_rt([lam], np.concatenate([[tau], dep[j + 1:]]),
                               n[j:], 1.0, ns)[1][0] - (cut - off[j] - e[j])
        dep[j] = brentq(f, 0.8 * t[j], 1.2 * t[j], xtol=1e-15, rtol=1e-15)
    return dep


def test_per_stop_offsets_of_a_drifting_monitor():
    """`signal_offset` as one value per stop: the simulated runs equal
    independent root finding with those offsets (1e-12 um); without
    noise, offsets of 1e-5 size move the layers by G o (relative 1e-3,
    the size of the second-order term), and `mean_t` is G o (1e-12); an
    array of equal values gives exactly what the number gives (the same
    arithmetic); wrong shapes and non-finite values are refused."""
    lam = 0.47
    drift = 4e-4 * np.array([3.0, 2.0, 1.0, 0.0])   # grows with time:
    #                       layer 3 is stopped first, layer 0 last
    e = np.random.default_rng(4).normal(0.0, 1e-3, (5, 4))
    got = ft.simulate_monitored_runs(lam, T4, N4, noise=e, n_sub=1.52,
                                     signal_offset=drift)
    for r in range(5):
        np.testing.assert_allclose(
            got[r], _brentq_run(lam, T4, N4, e[r], 1.52, drift),
            rtol=0, atol=1e-12)

    small = 1e-5 * np.array([1.0, -0.5, 2.0, 0.7])
    me = ft.monitoring_errors(lam, T4, N4, 1e-3, n_sub=1.52,
                              signal_offset=small)
    np.testing.assert_allclose(me.mean_t, me.propagation @ small,
                               rtol=1e-12, atol=0)
    run = ft.simulate_monitored_runs(lam, T4, N4, noise=np.zeros((1, 4)),
                                     n_sub=1.52, signal_offset=small)
    np.testing.assert_allclose(run[0] - T4, me.mean_t, rtol=1e-3)
    np.testing.assert_allclose(me.rel_mean_t, me.mean_t / T4, rtol=1e-14)
    np.testing.assert_allclose(me.twin().mu[:4], me.rel_mean_t, rtol=1e-14)

    a = ft.monitoring_errors(lam, T4, N4, 1e-3, n_sub=1.52,
                             signal_offset=2e-4)
    b = ft.monitoring_errors(lam, T4, N4, 1e-3, n_sub=1.52,
                             signal_offset=np.full(4, 2e-4))
    assert np.array_equal(a.mean_t, b.mean_t)
    assert np.array_equal(a.covariance, b.covariance)
    ra = ft.simulate_monitored_runs(lam, T4, N4, noise=e, n_sub=1.52,
                                    signal_offset=2e-4)
    rb = ft.simulate_monitored_runs(lam, T4, N4, noise=e, n_sub=1.52,
                                    signal_offset=np.full(4, 2e-4))
    assert np.array_equal(ra, rb)

    with pytest.raises(ValueError, match=r"\(4,\) array"):
        ft.monitoring_errors(lam, T4, N4, 1e-3, signal_offset=np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        ft.simulate_monitored_runs(lam, T4, N4, 1e-3, 2,
                                   signal_offset=[0.0, np.inf, 0.0, 0.0])
