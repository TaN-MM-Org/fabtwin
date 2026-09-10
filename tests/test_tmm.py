"""Solver anchors: closed-form Fresnel, absentee and quarter-wave
layers, exact energy conservation, the analytic Brewster zero, the
normal-incidence s = p identity, refusals, and cross-validation
against the independent open-source `tmm` reference."""
import numpy as np
import pytest

import fabtwin as ft

LAM = np.linspace(0.40, 0.80, 161)
S = ft.dispersion_shape(LAM, 0.550)
NSUB = ft.SIO2_MALITSON1965.n(LAM)


def _random_stack(rng, n_layers=20):
    t = rng.uniform(0.020, 0.120, n_layers)
    n0 = rng.uniform(1.6, 2.4, n_layers)
    return t, n0[:, None] * S[None, :]


def test_bare_interface_is_fresnel_exactly():
    R, T = ft.stack_rt(LAM, np.empty(0), np.empty((0, LAM.size)),
                       1.0, NSUB)
    T_fresnel = 4.0 * NSUB / (1.0 + NSUB) ** 2
    assert np.abs(T - T_fresnel).max() == 0.0
    assert np.abs(R + T - 1.0).max() < 1e-14


def test_half_wave_layer_is_absentee_exactly():
    """A half-wave layer contributes M = -I: the stack transmits
    exactly as the bare interface, whatever its index."""
    lam0 = 0.550
    for n_f in (1.7, 2.0, 2.35):
        Th = ft.transmittance(np.array([lam0]), np.array([lam0 / (2 * n_f)]),
                              np.array([[n_f]]), 1.0, np.array([1.46]))
        Tb = 4.0 * 1.46 / 2.46 ** 2
        assert abs(Th[0] - Tb) < 1e-14


def test_quarter_wave_antireflection_is_perfect_exactly():
    """n_f = sqrt(n_inc n_sub): the quarter-wave admittance transform
    gives T(lam0) = 1 exactly."""
    lam0 = 0.550
    Tq = ft.transmittance(np.array([lam0]), np.array([lam0 / 8.0]),
                          np.array([[2.0]]), 1.0, np.array([4.0]))
    assert abs(Tq[0] - 1.0) < 1e-14


def test_energy_conservation_lossless():
    rng = np.random.default_rng(1)
    for _ in range(3):
        t, nl = _random_stack(rng)
        R, T = ft.stack_rt(LAM, t, nl, 1.0, NSUB)
        assert np.abs(R + T - 1.0).max() < 1e-12


def test_absorbing_layers_absorb_and_gain_is_refused():
    rng = np.random.default_rng(2)
    t, nl = _random_stack(rng)
    R, T = ft.stack_rt(LAM, t, nl + 0.01j, 1.0, NSUB)
    A = 1.0 - R - T
    assert A.min() > 0.0
    with pytest.raises(ValueError):
        ft.stack_rt(LAM, t, nl - 0.01j, 1.0, NSUB)


def test_oblique_interface_fresnel_and_brewster():
    lam1 = np.array([0.6])
    n2 = 1.5
    th = np.deg2rad(40.0)
    ct1, st1 = np.cos(th), np.sin(th)
    ct2 = np.sqrt(1.0 - (st1 / n2) ** 2)
    for pol, r in (("s", (ct1 - n2 * ct2) / (ct1 + n2 * ct2)),
                   ("p", (n2 * ct1 - ct2) / (n2 * ct1 + ct2))):
        R, T = ft.stack_rt(lam1, np.empty(0), np.empty((0, 1)), 1.0,
                           np.array([n2]), th, pol)
        assert abs(R[0] - r ** 2) < 1e-14
        assert abs(R[0] + T[0] - 1.0) < 1e-14
    R_b, _ = ft.stack_rt(lam1, np.empty(0), np.empty((0, 1)), 1.0,
                         np.array([n2]), np.arctan(n2), "p")
    assert R_b[0] < 1e-30                    # the analytic Brewster zero


def test_s_equals_p_at_normal_incidence():
    rng = np.random.default_rng(3)
    t, nl = _random_stack(rng)
    Ts = ft.transmittance(LAM, t, nl, 1.0, NSUB, 0.0, "s")
    Tp = ft.transmittance(LAM, t, nl, 1.0, NSUB, 0.0, "p")
    assert np.abs(Ts - Tp).max() == 0.0


def test_against_the_open_tmm_reference():
    """The paper validates its solver to 5e-15 against S. J. Byrnes's
    open `tmm` package (arXiv:1603.02720); so does this one."""
    byrnes = pytest.importorskip("tmm")
    rng = np.random.default_rng(4)
    errs = []
    for _ in range(3):
        t = rng.uniform(0.020, 0.120, 8)
        n0 = rng.uniform(1.6, 2.4, 8)
        for lam in (0.45, 0.55, 0.70):
            res = byrnes.coh_tmm(
                "s", [1.0] + list(n0) + [1.46],
                [np.inf] + list(t * 1000) + [np.inf], 0.0, lam * 1000)
            T = ft.transmittance(np.array([lam]), t,
                                 n0[:, None] * np.ones((1, 1)), 1.0,
                                 np.array([1.46]))
            errs.append(abs(res["T"] - T[0]))
    assert max(errs) < 1e-12


def test_merit_weights_and_refusals():
    w, c0 = ft.notch_weights(LAM, 0.532, 0.015, 0.030)
    # a perfect notch (T = 1 pass, 0 stop) scores exactly 1; the
    # inverted filter exactly 0
    stop = np.abs(w == w) & (w < 0)
    T_perfect = (w >= 0).astype(float)
    T_perfect[w == 0.0] = 1.0
    assert abs(ft.merit(np.where(w < 0, 0.0, 1.0), w, c0) - 1.0) < 1e-14
    assert abs(ft.merit(np.where(w < 0, 1.0, 0.0), w, c0) - 0.0) < 1e-14
    wb, cb = ft.bandpass_weights(LAM, 0.58, 0.62, 0.03)
    assert abs(ft.merit(np.where(wb < 0, 0.0, 1.0), wb, cb) - 1.0) < 1e-14
    with pytest.raises(ValueError):
        ft.notch_weights(LAM, 0.532, 0.5, 0.030)     # empty pass band
    with pytest.raises(ValueError):
        ft.bandpass_weights(LAM, 0.1, 0.2, 0.01)     # off-grid band


def test_material_range_refusal_and_shape_identity():
    with pytest.raises(ValueError):
        ft.SI3N4_LUKE2015.n(np.array([0.2]))
    with pytest.raises(ValueError):
        ft.SIO2_MALITSON1965.n(np.array([10.0]))
    assert float(ft.dispersion_shape(np.array([0.550]), 0.550)[0]) == 1.0
    # engine reproduces the closed-form Sellmeier evaluation exactly
    lam = np.array([0.532])
    n2 = 1.0 + sum(c1 * lam ** 2 / (lam ** 2 - c2 ** 2)
                   for c1, c2 in ft.SI3N4_LUKE2015.terms)
    assert abs(ft.SI3N4_LUKE2015.n(lam)[0] - np.sqrt(n2[0])) == 0.0
