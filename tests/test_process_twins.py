"""Process and twin anchors: the exact deterministic corruption map,
the AR(1) stationary statistics, the centered log-normal, the flake
rate, the exact trace round trip, and Gaussian-twin parameter
recovery."""
import dataclasses

import numpy as np
import pytest

import fabtwin as ft

T_REC = np.linspace(0.030, 0.100, 20)
N_REC = np.linspace(1.70, 2.30, 20)


def test_deterministic_map_is_exact_with_randomness_off():
    proc = dataclasses.replace(ft.PAPER_PROCESS, sig_t=0.0, sig_n=0.0,
                               p_flake=0.0)
    tf, nf = proc.corrupt(T_REC, N_REC, np.random.default_rng(0))
    td, nd = proc.deterministic_map(T_REC, N_REC)
    assert np.abs(tf - td).max() == 0.0
    assert np.abs(nf - nd).max() == 0.0
    # the closed form itself: bias, drift + intermixing recursion
    assert np.abs(td - T_REC * 1.02).max() == 0.0
    grid = np.arange(20) / 19.0 - 0.5
    n_d = N_REC + proc.a_n * grid
    n_prev = np.concatenate([[n_d[0]], n_d[:-1]])
    assert np.abs(nd - (n_d + proc.kappa * (n_prev - n_d))).max() == 0.0


def test_ar1_noise_statistics():
    """At d = d_ref the AR(1) noise is stationary with std sig_t and
    lag-1 correlation rho -- the innovation scaling is exact, so the
    sample statistics must land within Monte-Carlo error."""
    proc = dataclasses.replace(ft.PAPER_PROCESS, p_flake=0.0, sig_n=0.0)
    t1 = np.full(20, proc.d_ref_um)
    D, _ = proc.ensemble(t1, N_REC, 4000, np.random.default_rng(2))
    eps = D / (t1 * (1.0 + proc.beta_t)) - 1.0
    assert abs(eps.std(axis=0).mean() - proc.sig_t) < 0.002
    r = np.corrcoef(eps[:, 5], eps[:, 6])[0, 1]
    assert abs(r - proc.rho) < 0.05


def test_lognormal_index_noise_is_centered():
    """The skew factor is normalized by exp(sig_n^2/2), so the mean
    fabricated index equals the deterministic map exactly in
    expectation."""
    proc = dataclasses.replace(ft.PAPER_PROCESS, p_flake=0.0, sig_t=0.0)
    _, Nn = proc.ensemble(T_REC, N_REC, 20000, np.random.default_rng(3))
    _, nd = proc.deterministic_map(T_REC, N_REC)
    assert np.abs(Nn.mean(axis=0) - nd).max() < 2e-3
    # and it is right-skewed
    dev = Nn / nd - 1.0
    from scipy.stats import skew
    assert skew(dev.ravel()) > 0.02


def test_flake_rate():
    proc = dataclasses.replace(ft.PAPER_PROCESS, sig_t=0.0, sig_n=0.0)
    D, _ = proc.ensemble(T_REC, N_REC, 20000, np.random.default_rng(4))
    flaked = np.any(np.abs(D / (T_REC * 1.02) - 1.0) > 1e-9, axis=1)
    assert abs(flaked.mean() - proc.p_flake) < 0.006


def test_process_parameter_refusals():
    with pytest.raises(ValueError):
        ft.DepositionProcess(rho=1.0)
    with pytest.raises(ValueError):
        ft.DepositionProcess(sig_t=-0.1)
    with pytest.raises(ValueError):
        ft.DepositionProcess(p_flake=1.5)


def test_error_parameterization_round_trip_is_exact():
    rng = np.random.default_rng(5)
    rt = np.tile(T_REC, (7, 1))
    rn = np.tile(N_REC, (7, 1))
    ftd = np.empty_like(rt)
    fnd = np.empty_like(rn)
    for m in range(7):
        ftd[m], fnd[m] = ft.PAPER_PROCESS.corrupt(rt[m], rn[m], rng)
    x = ft.errors_from_traces(rt, rn, ftd, fnd)
    t2, n2 = ft.apply_errors(T_REC, N_REC, x)
    assert np.abs(t2 - ftd).max() < 1e-15
    assert np.abs(n2 - fnd).max() < 1e-15


def test_gaussian_twin_recovers_known_parameters():
    rng = np.random.default_rng(6)
    dim = 12
    mu = rng.normal(0.0, 0.01, dim)
    A = rng.normal(size=(dim, dim)) * 0.003
    cov = A @ A.T + 0.001 ** 2 * np.eye(dim)
    L = np.linalg.cholesky(cov)
    x = mu + rng.normal(size=(60000, dim)) @ L.T
    tw = ft.GaussianTwin(x)
    assert np.abs(tw.mu - mu).max() < 3e-4
    assert np.abs(tw.cov - cov).max() < 5e-6
    # reparameterization is exact: transform(0) == mu
    assert np.abs(tw.transform(np.zeros((1, dim)))[0] - tw.mu).max() == 0.0
    # diagonal variant zeroes the off-diagonals
    twd = ft.GaussianTwin(x, diagonal=True)
    off = twd.cov - np.diag(np.diag(twd.cov))
    assert np.abs(off).max() == 0.0


def test_trace_dataset_shapes_and_exposure():
    rng = np.random.default_rng(7)
    rec_t = np.tile(T_REC, (5, 1))
    rec_n = np.tile(N_REC, (5, 1))
    rt, rn, ftd, fnd = ft.PAPER_PROCESS.trace_dataset(rec_t, rec_n, 3, rng)
    assert rt.shape == (15, 20) and fnd.shape == (15, 20)
    assert np.abs(rt[0] - rt[1]).max() == 0.0     # runs share the recipe
