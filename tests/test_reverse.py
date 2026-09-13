"""v0.4 reverse-engineering anchors: noise-free round trip through
the exact physics, exact zero at a perfect deposition, uncertainty
calibration on seeded noise, and the three reliability refusals --
underdetermined, exactly degenerate (adjacent same-index layers), and
practically driven by the Tikhonravov reliability analysis."""
import numpy as np
import pytest

import fabtwin as ft
from fabtwin import errors_from_spectrum

LAM = np.linspace(0.40, 0.80, 201)
S = ft.dispersion_shape(LAM, 0.550)
NSUB = ft.SIO2_MALITSON1965.n(LAM)


def _recipe(n_layers=6, seed=1):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.050, 0.120, n_layers)
    n0 = np.empty(n_layers)
    n0[0::2] = 1.46                              # alternating L/H,
    n0[1::2] = 2.0                               # H against the substrate
    return t, n0


def _spectrum(t, n0):
    T, _, _ = ft.transmittance_and_grads(LAM, t, n0, S, n_sub=NSUB)
    return T


def test_noise_free_round_trip_and_exact_zero():
    t0, n0 = _recipe()
    x_true = np.array([0.021, -0.034, 0.012, 0.040, -0.008, 0.017])
    rec = errors_from_spectrum(LAM, _spectrum(t0 * (1 + x_true), n0),
                               t0, n0, S, n_sub=NSUB)
    assert np.max(np.abs(rec.dt_over_t - x_true)) < 1e-6
    assert np.max(np.abs(rec.t_um - t0 * (1 + x_true))) < 1e-6
    assert rec.chi2 < 1e-12 and rec.dof == LAM.size - 6
    # perfect deposition recovers exactly zero errors
    rec0 = errors_from_spectrum(LAM, _spectrum(t0, n0), t0, n0, S,
                                n_sub=NSUB)
    assert np.max(np.abs(rec0.dt_over_t)) < 1e-8


def test_uncertainty_calibration_on_seeded_noise():
    """With known measurement noise, the recovered errors must sit
    within their reported sigmas (|z| bounded), and the sigmas must
    not be wildly conservative."""
    t0, n0 = _recipe()
    x_true = np.array([0.015, -0.022, 0.030, -0.011, 0.006, -0.028])
    rng = np.random.default_rng(7)
    sig = 2e-3
    Tm = np.clip(_spectrum(t0 * (1 + x_true), n0)
                 + rng.normal(0.0, sig, LAM.size), 0.0, 1.0)
    rec = errors_from_spectrum(LAM, Tm, t0, n0, S, n_sub=NSUB,
                               sigma_T=sig)
    z = (rec.dt_over_t - x_true) / rec.sigma
    assert np.max(np.abs(z)) < 4.0               # calibrated, seeded
    assert np.all(rec.sigma < 0.05)              # and informative
    # chi2 consistent with dof for correctly stated noise
    assert 0.5 * rec.dof < rec.chi2 < 1.7 * rec.dof


def test_underdetermined_and_degenerate_refusals():
    t0, n0 = _recipe()
    lam5 = LAM[::50]                             # 5 points, 6 layers
    with pytest.raises(ValueError):
        errors_from_spectrum(lam5, np.full(lam5.size, 0.5), t0, n0,
                             ft.dispersion_shape(lam5, 0.550),
                             n_sub=ft.SIO2_MALITSON1965.n(lam5))
    # adjacent layers of the SAME index: only their SUM enters the
    # transfer matrix, the difference is invisible to any spectrum
    t_deg = np.array([0.080, 0.060, 0.100])
    n_deg = np.array([2.0, 2.0, 1.46])
    T = ft.transmittance_and_grads(LAM, t_deg, n_deg, S, n_sub=NSUB)[0]
    with pytest.raises(ValueError, match="non-unique|singular"):
        errors_from_spectrum(LAM, T, t_deg, n_deg, S, n_sub=NSUB)


def test_input_validation():
    t0, n0 = _recipe()
    T = _spectrum(t0, n0)
    with pytest.raises(ValueError):
        errors_from_spectrum(LAM, T + 2.0, t0, n0, S, n_sub=NSUB)
    with pytest.raises(ValueError):
        errors_from_spectrum(LAM, T, -t0, n0, S, n_sub=NSUB)
    with pytest.raises(ValueError):
        errors_from_spectrum(LAM, T, t0, n0, S, n_sub=NSUB, sigma_T=0.0)
    with pytest.raises(ValueError):
        errors_from_spectrum(LAM, T[:-1], t0, n0, S, n_sub=NSUB)
