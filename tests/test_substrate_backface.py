"""Thick substrate with a bare back face (0.8.0): forward optics,
exact gradients, the design model, and spectrum recovery; plus the
bound flag and chi-square p-value of the recoveries."""
import math
import warnings

import numpy as np
import pytest

import fabtwin as ft
from fabtwin.gradients import stack_rta_and_grads
from fabtwin.merits import (LinearMerit, OpticalModel, SpecMarginMerit,
                            model_merit_and_grad, model_spectra)


def _inc_tmm(byrnes, pol, lam, t, n, ninc, nsub, nexit, th):
    """The independent `tmm` package: coherent layers between an
    incoherent incidence medium and an incoherent (thick) substrate."""
    def one(p):
        r = byrnes.inc_tmm(p, [ninc] + list(n) + [nsub, nexit],
                           [np.inf] + list(np.asarray(t) * 1000)
                           + [1e6, np.inf],
                           ["i"] + ["c"] * len(t) + ["i", "i"], th,
                           lam * 1000)
        return r["R"], r["T"]
    if pol == "u":
        (Rs, Ts), (Rp, Tp) = one("s"), one("p")
        return 0.5 * (Rs + Rp), 0.5 * (Ts + Tp)
    return one(pol)


def test_thick_substrate_matches_the_open_tmm_incoherent_solver():
    byrnes = pytest.importorskip("tmm")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(150):
        N = int(rng.integers(0, 7))
        t = rng.uniform(0.02, 0.2, N)
        n = rng.uniform(1.3, 2.5, N) + 1j * rng.uniform(0, 0.3, N) * \
            (rng.random(N) < 0.5)
        ninc = float(rng.choice([1.0, 1.33]))
        nsub = float(rng.choice([1.46, 1.52, 1.8]))
        nexit = float(rng.choice([1.0, 1.33]))
        th = rng.uniform(0, 1.4)
        pol = str(rng.choice(["s", "p", "u"]))
        lam = rng.uniform(0.4, 0.9)
        R, T = ft.stack_rt(np.array([lam]), t, n[:, None], ninc, nsub, th,
                           pol, n_exit=nexit)
        Rb, Tb = _inc_tmm(byrnes, pol, lam, t, n, ninc, nsub, nexit, th)
        worst = max(worst, abs(R[0] - Rb), abs(T[0] - Tb))
    assert worst < 1e-12
    # a dispersive substrate given as an (L,) array, in one call
    lam = np.linspace(0.4, 0.8, 9)
    nsub = ft.SIO2_MALITSON1965.n(lam)
    t = np.array([0.09, 0.07, 0.012])
    n = np.array([2.05, 1.46, 0.5 + 2.5j])
    R, T = ft.stack_rt(lam, t, n, 1.0, nsub, 0.5, "u", n_exit=1.0)
    for j in range(lam.size):
        Rb, Tb = _inc_tmm(byrnes, "u", lam[j], t, n, 1.0, nsub[j], 1.0, 0.5)
        assert abs(R[j] - Rb) < 1e-12 and abs(T[j] - Tb) < 1e-12


def test_bare_and_perfectly_coated_plate_closed_forms():
    lam = np.array([0.5, 0.6])
    for ns in (1.46, 1.52, 2.0):
        R1 = ((ns - 1) / (ns + 1)) ** 2
        # bare plate: T = (1 - R1) / (1 + R1) = 2 n / (n^2 + 1)
        R, T = ft.stack_rt(lam, np.zeros(0), np.zeros((0, 2)), 1.0, ns,
                           n_exit=1.0)
        assert np.abs(T - 2 * ns / (ns ** 2 + 1)).max() < 1e-15
        assert np.abs(R - 2 * R1 / (1 + R1)).max() < 1e-15
        # ideal quarter-wave AR coating on the front at 0.5 um: only the
        # back face reflects, T = 1 - R1 = 4 n / (n + 1)^2
        nf = np.sqrt(ns)
        R, T = ft.stack_rt(lam[:1], np.array([0.5 / (4 * nf)]),
                           np.array([nf]), 1.0, ns, n_exit=1.0)
        assert abs(T[0] - 4 * ns / (ns + 1) ** 2) < 1e-14
        assert abs(R[0] - R1) < 1e-14
    # lossless: R + T = 1 at any angle and polarization
    rng = np.random.default_rng(1)
    lam = np.linspace(0.4, 0.8, 11)
    for pol in ("s", "p", "u"):
        R, T = ft.stack_rt(lam, rng.uniform(0.03, 0.2, 6),
                           rng.uniform(1.4, 2.4, 6), 1.0, 1.52,
                           0.9, pol, n_exit=1.0)
        assert np.abs(R + T - 1).max() < 1e-14


def test_thick_substrate_gradients_match_finite_differences():
    rng = np.random.default_rng(2)
    lam = np.linspace(0.45, 0.8, 6)
    nsub = ft.SIO2_MALITSON1965.n(lam)
    for _ in range(12):
        N = int(rng.integers(1, 6))
        t = rng.uniform(0.03, 0.2, N)
        n = (rng.uniform(1.3, 2.5, (N, 1)) * np.ones((1, lam.size))
             + 1j * rng.uniform(0.01, 0.2, (N, 1)))
        th = rng.uniform(0, 1.3)
        pol = str(rng.choice(["s", "p", "u"]))
        nexit = float(rng.choice([1.0, 1.33]))
        g = stack_rta_and_grads(lam, t, n, 1.0, nsub, th, pol,
                                n_exit=nexit)

        def fwd(tt, nn):
            R, T = ft.stack_rt(lam, tt, nn, 1.0, nsub, th, pol,
                               n_exit=nexit)
            return dict(R=R, T=T, A=1 - R - T)

        base = fwd(t, n)
        for q in "RTA":
            assert np.abs(g[q] - base[q]).max() < 1e-13
        h = 1e-6
        for i in range(N):
            e = np.zeros(N)
            e[i] = h
            E = np.zeros((N, 1))
            E[i] = h
            for p, (a, b) in (("t", (fwd(t + e, n), fwd(t - e, n))),
                              ("n", (fwd(t, n + E), fwd(t, n - E))),
                              ("k", (fwd(t, n + 1j * E),
                                     fwd(t, n - 1j * E)))):
                for q in "RTA":
                    fd = (a[q] - b[q]) / (2 * h)
                    ex = g[f"d{q}_d{p}"][i]
                    assert np.all(np.abs(ex - fd)
                                  <= 1e-7 * np.abs(fd) + 1e-8), (p, q)


def test_model_path_carries_the_back_face():
    lam = np.linspace(0.45, 0.7, 26)
    S = ft.dispersion_shape(lam, 0.55)
    rng = np.random.default_rng(3)
    t, n0 = rng.uniform(0.03, 0.15, 8), rng.uniform(1.6, 2.3, 8)
    w, c = ft.notch_weights(lam, 0.55, 0.02, 0.02)
    m = OpticalModel(LinearMerit(np.tile(w / 2, (2, 1)), c, "T"),
                     ((0.0, "s"), (0.3, "u")), n_exit=1.0)
    R, T, A = model_spectra(m, lam, t, n0, S, 1.0, 1.52)
    _, T1 = ft.stack_rt(lam, t, n0[:, None] * S, 1.0, 1.52, 0.3, "u",
                        n_exit=1.0)
    assert np.abs(T[1] - T1).max() < 1e-13      # two separate codes
    J, gt, gn = model_merit_and_grad(m, lam, t, n0, S, 1.0, 1.52)
    h = 1e-6
    for i in (0, 4, 7):
        e = np.zeros(8)
        e[i] = h
        fd_t = (model_merit_and_grad(m, lam, t + e, n0, S, 1.0, 1.52)[0]
                - model_merit_and_grad(m, lam, t - e, n0, S, 1.0, 1.52)[0]
                ) / (2 * h)
        fd_n = (model_merit_and_grad(m, lam, t, n0 + e, S, 1.0, 1.52)[0]
                - model_merit_and_grad(m, lam, t, n0 - e, S, 1.0, 1.52)[0]
                ) / (2 * h)
        assert abs(gt[i] - fd_t) < 1e-7 * abs(fd_t) + 1e-9
        assert abs(gn[i] - fd_n) < 1e-7 * abs(fd_n) + 1e-9
    # a bare glass plate in air cannot pass more than 2n/(n^2+1)
    stop = (lam > 0.54) & (lam < 0.56)
    pas = lam < 0.5
    spec = OpticalModel(SpecMarginMerit(stop, pas, 0.1, 0.93), n_exit=1.0)
    _, T, _ = model_spectra(spec, lam, np.array([0.1]), np.array([1.0]),
                            np.ones(lam.size), 1.0, 1.52)
    assert np.abs(T - 2 * 1.52 / (1.52 ** 2 + 1)).max() < 1e-15


def test_back_face_refusals():
    lam = np.array([0.5, 0.6])
    t, n = np.array([0.1]), np.array([2.0])
    with pytest.raises(ValueError, match="lossless"):
        ft.stack_rt(lam, t, n, 1.0, 1.5 + 0.01j, n_exit=1.0)
    with pytest.raises(ValueError, match="lossless"):
        ft.stack_rt(lam, t, n, 1.0, 1.5, n_exit=1.0 + 0.1j)
    with pytest.raises(ValueError, match="positive"):
        ft.stack_rt(lam, t, n, 1.0, 1.5, n_exit=0.0)
    with pytest.raises(ValueError, match="plays no role"):   # prism case
        ft.stack_rt(lam, t, n, 1.8, 1.5, 1.2, n_exit=1.0)
    with pytest.raises(ValueError, match="plays no role"):
        stack_rta_and_grads(lam, t, n, 1.8, 1.5, 1.2, "s", n_exit=1.0)
    with pytest.raises(ValueError, match="n_exit"):
        OpticalModel(LinearMerit(np.ones(2)), n_exit=-1.0)
    with pytest.raises(ValueError):
        ft.errors_from_spectrum(lam, np.array([0.5, 0.5]), t, n,
                                np.ones(2), n_sub=1.5 + 0.1j, n_exit=1.0)


def _plate_case():
    lam = np.linspace(0.40, 0.80, 121)
    S = ft.dispersion_shape(lam, 0.550)
    nsub = ft.SIO2_MALITSON1965.n(lam)
    t0 = np.array([0.070, 0.095, 0.060, 0.110, 0.080])
    n0 = np.array([2.2, 1.5, 2.2, 1.5, 2.2])
    xt = np.array([0.03, -0.02, 0.015, 0.01, -0.025])
    return lam, S, nsub, t0, n0, xt


def test_recovery_from_a_measured_plate_spectrum():
    """The spectrum is made by the independent tmm package (coherent
    coating, incoherent 1 mm substrate, air behind it)."""
    byrnes = pytest.importorskip("tmm")
    lam, S, nsub, t0, n0, xt = _plate_case()
    nl = n0[:, None] * S
    T = np.array([_inc_tmm(byrnes, "s", lam[j], t0 * (1 + xt), nl[:, j],
                           1.0, nsub[j], 1.0, 0.0)[1]
                  for j in range(lam.size)])
    rec = ft.errors_from_spectrum(lam, T, t0, n0, S, n_sub=nsub,
                                  n_exit=1.0, n_starts=4)
    assert np.abs(rec.dt_over_t - xt).max() < 1e-6
    assert not rec.at_bound.any()
    r2 = ft.errors_from_spectra(lam, [ft.Measurement(0.0, "s", "T", T)],
                                t0, n0, S, n_sub=nsub, n_exit=1.0,
                                n_starts=4)
    assert np.abs(r2.dt_over_t - rec.dt_over_t).max() < 1e-8
    # noisy data with the back face ignored: the layers are asked to
    # explain the ~4 % back-face loss; the answer is wrong, and the
    # chi-square check says so
    Tm = np.clip(T + np.random.default_rng(7).normal(0, 2e-3, lam.size),
                 0, 1)
    with pytest.warns(RuntimeWarning, match="does not describe"):
        bad = ft.errors_from_spectrum(lam, Tm, t0, n0, S, n_sub=nsub,
                                      sigma_T=2e-3, n_starts=4)
    assert bad.chi2_pvalue < 1e-6
    assert np.any(np.abs(bad.dt_over_t - xt) > 4 * bad.sigma)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        good = ft.errors_from_spectrum(lam, Tm, t0, n0, S, n_sub=nsub,
                                       sigma_T=2e-3, n_exit=1.0,
                                       n_starts=4)
    assert not good.at_bound.any() and good.chi2_pvalue > 1e-3
    assert np.all(np.abs(good.dt_over_t - xt) < 4 * good.sigma)


def test_joint_recovery_through_the_back_face():
    lam, S, nsub, t0, n0, xt = _plate_case()
    dn = np.array([0.02, -0.015, 0.01, 0.0, -0.02])
    nl = (n0 + dn)[:, None] * S
    ms = []
    for ang, pol in ((0.0, "s"), (45.0, "s"), (45.0, "p"), (60.0, "s"),
                     (60.0, "p")):
        _, T = ft.stack_rt(lam, t0 * (1 + xt), nl, 1.0, nsub,
                           np.deg2rad(ang), pol, n_exit=1.0)
        ms.append(ft.Measurement(np.deg2rad(ang), pol, "T", T))
    r = ft.errors_from_spectra(lam, ms, t0, n0, S, fit_index=True,
                               n_sub=nsub, n_exit=1.0, n_starts=3)
    assert np.abs(r.dt_over_t - xt).max() < 1e-6
    assert np.abs(r.dn - dn).max() < 1e-6
    assert not r.at_bound_t.any() and not r.at_bound_n.any()
    assert r.chi2_pvalue is None                 # no sigma given


def test_bound_flag_and_chi2_pvalue():
    lam, S, nsub, t0, n0, xt = _plate_case()
    x = xt.copy()
    x[1] = -0.09                     # beyond 2 * max_error = 0.06
    T, _, _ = ft.transmittance_and_grads(lam, t0 * (1 + x), n0, S,
                                         n_sub=nsub)
    Tm = np.clip(T + np.random.default_rng(1).normal(0, 1e-3, lam.size),
                 0, 1)
    with pytest.warns(RuntimeWarning, match="layer 2 thickness error"):
        rec = ft.errors_from_spectrum(lam, Tm, t0, n0, S, n_sub=nsub,
                                      sigma_T=1e-3, max_error=0.03,
                                      n_starts=3)
    assert rec.at_bound[1] and abs(rec.dt_over_t[1] + 0.06) < 1e-9
    # chi2 p-value against the closed form for an even number of
    # degrees of freedom k = 2m: exp(-x/2) sum_{j<m} (x/2)^j / j!
    lam2 = lam[:-2]                  # 119 points - 5 layers = 114, even
    T2, _, _ = ft.transmittance_and_grads(lam2, t0 * (1 + xt), n0, S[:-2],
                                          n_sub=nsub[:-2])
    Tm2 = np.clip(T2 + np.random.default_rng(2).normal(0, 1e-3,
                                                         lam2.size), 0, 1)
    rec = ft.errors_from_spectrum(lam2, Tm2, t0, n0, S[:-2],
                                  n_sub=nsub[:-2], sigma_T=1e-3,
                                  n_starts=3)
    assert rec.dof % 2 == 0
    h = rec.chi2 / 2
    ref = math.exp(-h) * sum(h ** j / math.factorial(j)
                             for j in range(rec.dof // 2))
    assert abs(rec.chi2_pvalue - ref) < 1e-12
    assert not rec.at_bound.any()
    assert ft.errors_from_spectrum(lam2, Tm2, t0, n0, S[:-2],
                                   n_sub=nsub[:-2],
                                   n_starts=3).chi2_pvalue is None
