"""Drift test, novelty p-values, group and drift-robust conformal
guarantees, and joint thickness-and-index recovery (0.7.0)."""
import dataclasses

import numpy as np
import pytest
from scipy.spatial.distance import cdist

import fabtwin as ft
from fabtwin.conformal import AdaptiveConformal, mondrian_quantiles
from fabtwin.fidelity import drift_test, energy_distance, novelty_pvalues
from fabtwin.gradients import stack_rta_and_grads
from fabtwin.reverse import Measurement, errors_from_spectra


def _runs(proc, K, seed, t=None, n=None):
    t = np.array([0.07, 0.09, 0.06, 0.11]) if t is None else t
    n = np.array([2.1, 1.6, 2.1, 1.6]) if n is None else n
    rng = np.random.default_rng(seed)
    ft_, fn = proc.ensemble(t, n, K, rng)
    return ft.errors_from_traces(np.tile(t, (K, 1)), np.tile(n, (K, 1)),
                                 ft_, fn)


def test_energy_distance_matches_a_direct_computation():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(30, 5))
    b = rng.normal(size=(40, 5)) + 0.3
    direct = (2 * cdist(a, b).mean() - cdist(a, a).mean()
              - cdist(b, b).mean())
    assert energy_distance(a, b) == pytest.approx(direct, rel=1e-10)
    assert energy_distance(a, a) == pytest.approx(0.0, abs=1e-12)


def test_drift_test_size_and_power():
    # same process: p-values are (close to) uniform, so few are small
    ps = [drift_test(_runs(ft.PAPER_PROCESS, 30, s),
                     _runs(ft.PAPER_PROCESS, 30, 100 + s),
                     n_perm=99, seed=s)["p_value"] for s in range(40)]
    assert np.mean(np.array(ps) <= 0.05) <= 0.15
    # a drifted machine (thickness bias 2 % -> 4 %) is detected
    drifted = dataclasses.replace(ft.PAPER_PROCESS, beta_t=0.04)
    r = drift_test(_runs(ft.PAPER_PROCESS, 30, 1), _runs(drifted, 30, 2),
                   n_perm=199, seed=0)
    assert r["p_value"] <= 0.01


def test_novelty_pvalues_are_valid_and_flag_unseen_errors():
    x = _runs(ft.PAPER_PROCESS, 900, 3)
    tr, cal, new = x[:300], x[300:600], x[600:]
    p = novelty_pvalues(tr, cal, new)
    for u in (0.05, 0.1, 0.2):          # P(p <= u) <= u, up to sampling
        assert np.mean(p <= u) <= u + 3 * np.sqrt(u * (1 - u) / p.size)
    odd = new[:5].copy()
    odd[:, 0] += 0.2                    # a 20 % error on layer 1, never seen
    assert np.all(novelty_pvalues(tr, cal, odd) <= 1.0 / 301 + 1e-12)


def test_mondrian_quantiles_cover_each_group():
    rng = np.random.default_rng(4)
    alpha, n = 0.1, 39
    hits = {0: [], 1: []}
    for _ in range(1500):
        cal = np.concatenate([np.abs(rng.normal(0, 1, n)),
                              np.abs(rng.normal(0, 5, n))])
        grp = np.repeat([0, 1], n)
        q = mondrian_quantiles(cal, grp, alpha)
        hits[0].append(abs(rng.normal(0, 1)) <= q[0])
        hits[1].append(abs(rng.normal(0, 5)) <= q[1])
    exact = ft.conformal_coverage_exact(n, alpha)
    for g in (0, 1):
        m = np.mean(hits[g])
        assert abs(m - exact) < 4 * np.sqrt(exact * (1 - exact) / 1500)
    # a single pooled quantile would under-cover the wide group
    with pytest.raises(ValueError, match="group"):
        mondrian_quantiles(np.ones(5), [0, 0, 0, 1, 1], 0.1)


def test_adaptive_conformal_bound_holds_under_drift():
    rng = np.random.default_rng(5)
    aci = AdaptiveConformal(np.abs(rng.normal(size=100)), alpha=0.1,
                            gamma=0.01, window=100)
    T = 3000
    for t in range(T):
        scale = 1.0 + 2.0 * (t > 1000) + 0.002 * t     # a jump, then a ramp
        aci.update(abs(rng.normal(0, scale)))
        # the deterministic long-run bound, at every step
        assert abs(aci.miss_rate() - 0.1) <= aci.bound() + 1e-12
    assert abs(aci.miss_rate() - 0.1) < 0.02
    # a fixed split-conformal quantile from the start would fail badly
    q0 = ft.conformal_quantile(np.abs(rng.normal(size=100)), 0.1)
    miss = np.mean([abs(rng.normal(0, 1 + 2 * (t > 1000) + 0.002 * t)) > q0
                    for t in range(T)])
    assert miss > 0.3


# --- joint thickness and index recovery -----------------------------------

LAM = np.linspace(0.40, 0.90, 121)
S = ft.dispersion_shape(LAM, 0.55)
T0 = np.array([0.070, 0.095, 0.060, 0.110, 0.080])
N0 = np.array([2.2, 1.5, 2.2, 1.5, 2.2])
XT = np.array([0.03, -0.02, 0.015, 0.01, -0.025])
DN = np.array([0.02, -0.015, 0.01, 0.0, -0.02])


def _meas(th, pol, q, rng=None, sig=0.002):
    g = stack_rta_and_grads(LAM, T0 * (1 + XT), (N0 + DN)[:, None] * S,
                            1.0, 1.46, th, pol)
    v = g[q] if rng is None else np.clip(
        g[q] + rng.normal(0, sig, LAM.size), 0, 1)
    return Measurement(th, pol, q, v, sig)


ANGLES = [(a, p) for a in (np.pi / 4, np.pi / 3) for p in "sp"]


def test_joint_recovery_is_exact_without_noise():
    ms = [_meas(0, "s", "T")] + [_meas(a, p, "T") for a, p in ANGLES]
    r = errors_from_spectra(LAM, ms, T0, N0, S, fit_index=True, n_sub=1.46)
    assert np.abs(r.dt_over_t - XT).max() < 1e-6
    assert np.abs(r.dn - DN).max() < 1e-6


def test_one_normal_spectrum_cannot_give_index_and_thickness():
    rng = np.random.default_rng(1)
    for ms in ([_meas(0, "s", "T", rng)],
               [_meas(0, "s", "T", rng), _meas(0, "s", "R", rng)]):
        with pytest.raises(ValueError):
            errors_from_spectra(LAM, ms, T0, N0, S, fit_index=True,
                                n_sub=1.46, n_starts=4)


def test_several_angles_recover_both_within_their_error_bars():
    rng = np.random.default_rng(1)
    ms = [_meas(0, "s", "T", rng)] + [_meas(a, p, "T", rng)
                                      for a, p in ANGLES]
    r = errors_from_spectra(LAM, ms, T0, N0, S, fit_index=True, n_sub=1.46,
                            n_boot=30)
    assert np.all(np.abs(r.dt_over_t - XT) < 4 * r.sigma_t)
    assert np.all(np.abs(r.dn - DN) < 4 * r.sigma_n)
    assert 0.8 < r.chi2 / r.dof < 1.25
    # the bootstrap and the linear estimate agree within a factor 2
    ratio = np.concatenate([r.boot_sigma_t / r.sigma_t,
                            r.boot_sigma_n / r.sigma_n])
    assert np.all((ratio > 0.5) & (ratio < 2.0))


def test_thickness_only_matches_the_original_function():
    ms = [_meas(0, "s", "T")]
    g = stack_rta_and_grads(LAM, T0 * (1 + XT), N0[:, None] * S, 1.0, 1.46,
                            0.0, "s")
    ms = [Measurement(0.0, "s", "T", g["T"], 0.002)]
    r = errors_from_spectra(LAM, ms, T0, N0, S, n_sub=1.46)
    r0 = ft.errors_from_spectrum(LAM, g["T"], T0, N0, S, n_sub=1.46,
                                 sigma_T=0.002)
    assert np.abs(r.dt_over_t - r0.dt_over_t).max() < 1e-8
    assert np.allclose(r.sigma_t, r0.sigma, rtol=1e-6)
    assert np.all(r.dn == 0)


def test_joint_refusals():
    with pytest.raises(ValueError):
        errors_from_spectra(LAM, [], T0, N0, S)
    with pytest.raises(ValueError):
        errors_from_spectra(LAM, [Measurement(0, "s", "A", np.zeros(121))],
                            T0, N0, S)
    with pytest.raises(ValueError):
        errors_from_spectra(LAM[:6], [Measurement(0, "s", "T",
                                                  np.full(6, 0.5))],
                            T0, N0, S[:6], fit_index=True)
