"""Risk, inverse-design and robustification anchors: hand-exact CVaR,
the closed-form quarter-wave recovery by the probe-seeded engine, and
the frozen-latent pathwise CVaR gradient against central finite
differences -- the whole differentiable fabrication loop held to
deterministic checks."""
import numpy as np
import pytest

import fabtwin as ft


# ------------------------------ risk ------------------------------


def test_cvar_is_exact_on_hand_sets():
    v = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    assert ft.cvar(v, 0.4) == 1.5          # mean of the 2 smallest
    assert ft.cvar(v, 1.0) == 3.0          # the mean
    assert ft.cvar(v, 0.01) == 1.0         # the minimum
    with pytest.raises(ValueError):
        ft.cvar(v, 0.0)
    with pytest.raises(ValueError):
        ft.cvar(np.empty(0), 0.5)
    s = ft.tail_statistics(v, 0.4)
    assert s["CVaR"] == 1.5 and s["mean"] == 3.0


def test_pass_fail_spec():
    stop = np.zeros(10, bool)
    stop[4:6] = True
    pas = ~stop
    T_good = np.where(stop, 0.05, 0.95)
    T_leaky = np.where(stop, 0.20, 0.95)
    T_dim = np.where(stop, 0.05, 0.80)
    T = np.stack([T_good, T_leaky, T_dim])
    ok = ft.pass_fail(T, stop, pas)
    assert list(ok) == [True, False, False]
    assert ft.yield_fraction(T, stop, pas) == pytest.approx(1.0 / 3.0)


# -------------------------- inverse design -------------------------


def test_inverse_design_recovers_the_quarter_wave_closed_form():
    """Single-layer antireflection on an n = 4 substrate: the unique
    in-box optimum is n = sqrt(4) = 2, t = lam0/8, T = 1. The engine
    must find it from random probes."""
    lam0 = 0.550
    lam = np.array([lam0])
    box = ft.DesignBox(1, 0.020, 0.120, 1.6, 2.4)
    t, n0, J, traj = ft.inverse_design(
        lam, np.ones(1), np.array([1.0]), 0.0, box, n_probe=50,
        n_seed=3, n_iter=300, lr=8e-3, n_sub=np.array([4.0]))
    assert abs(J - 1.0) < 1e-10
    assert abs(t[0] - lam0 / 8.0) < 1e-6
    assert abs(n0[0] - 2.0) < 1e-6
    assert np.all(np.diff(traj) >= 0.0)     # best-so-far is monotone


def test_engine_beats_equal_budget_random_search_on_the_notch():
    lam = np.linspace(0.45, 0.65, 41)
    S = ft.dispersion_shape(lam, 0.550)
    nsub = ft.SIO2_MALITSON1965.n(lam)
    w, c0 = ft.notch_weights(lam, 0.532, 0.015, 0.030)
    box = ft.DesignBox(8, 0.020, 0.120, 1.6, 2.4)
    t, n0, J, traj = ft.inverse_design(lam, S, w, c0, box, n_probe=60,
                                       n_seed=2, n_iter=30,
                                       n_sub=nsub, seed=0)
    budget = len(traj)
    _, _, J_rs, _ = ft.random_search(lam, S, w, c0, box, budget,
                                     n_sub=nsub, seed=1)
    assert J > J_rs


def test_design_box_refusals():
    with pytest.raises(ValueError):
        ft.DesignBox(0, 0.02, 0.12, 1.6, 2.4)
    with pytest.raises(ValueError):
        ft.DesignBox(5, 0.12, 0.02, 1.6, 2.4)


# ------------------------- robustification -------------------------


def _setup_robust():
    rng = np.random.default_rng(0)
    lam = np.linspace(0.40, 0.80, 81)
    S = ft.dispersion_shape(lam, 0.550)
    nsub = ft.SIO2_MALITSON1965.n(lam)
    w, c0 = ft.notch_weights(lam, 0.532, 0.015, 0.030)
    box = ft.DesignBox(8, 0.020, 0.120, 1.6, 2.4)
    t = rng.uniform(0.030, 0.110, 8)
    n0 = rng.uniform(1.7, 2.3, 8)
    tw = ft.GaussianTwin(rng.normal(0.0, 0.01, (300, 16)))
    return lam, S, nsub, w, c0, box, t, n0, tw, rng


def test_frozen_latent_cvar_gradient_matches_finite_differences():
    """With the latent batch frozen the empirical CVaR is a
    deterministic function of the design; its hand-chained gradient
    through twin and adjoint must match central differences."""
    lam, S, nsub, w, c0, box, t, n0, tw, rng = _setup_robust()
    z = rng.normal(size=(64, 16))
    _, gt, gn, _ = ft.cvar_objective_and_grad(lam, t, n0, S, w, c0, tw,
                                              z, 0.1, box, n_sub=nsub)
    h = 1e-6

    def f(tt, nn):
        return ft.cvar_objective_and_grad(lam, tt, nn, S, w, c0, tw, z,
                                          0.1, box, n_sub=nsub)[0]

    for i in range(8):
        tp, tm = t.copy(), t.copy()
        tp[i] += h
        tm[i] -= h
        fd = (f(tp, n0) - f(tm, n0)) / (2 * h)
        assert abs(gt[i] - fd) / max(abs(fd), 1e-10) < 1e-6
        np_, nm = n0.copy(), n0.copy()
        np_[i] += h
        nm[i] -= h
        fd = (f(t, np_) - f(t, nm)) / (2 * h)
        assert abs(gn[i] - fd) / max(abs(fd), 1e-10) < 1e-6


def test_frozen_latent_mean_variance_gradient_matches_fd():
    lam, S, nsub, w, c0, box, t, n0, tw, rng = _setup_robust()
    z = rng.normal(size=(48, 16))
    kw = dict(n_sub=nsub, mean_variance=True, beta=1.5)
    _, gt, gn, _ = ft.cvar_objective_and_grad(lam, t, n0, S, w, c0, tw,
                                              z, 0.1, box, **kw)
    h = 1e-6

    def f(tt, nn):
        return ft.cvar_objective_and_grad(lam, tt, nn, S, w, c0, tw, z,
                                          0.1, box, **kw)[0]

    for i in (0, 3, 7):
        tp, tm = t.copy(), t.copy()
        tp[i] += h
        tm[i] -= h
        fd = (f(tp, n0) - f(tm, n0)) / (2 * h)
        assert abs(gt[i] - fd) / max(abs(fd), 1e-10) < 1e-6


def test_robustify_raises_the_twin_cvar():
    """Ascending the empirical CVaR under the twin must raise the
    twin-evaluated CVaR of the design (fresh evaluation draws)."""
    lam, S, nsub, w, c0, box, t, n0, tw, rng = _setup_robust()

    def twin_cvar(tt, nn, seed):
        r = np.random.default_rng(seed)
        out = ft.evaluate_under_process(
            lambda a, b, K: tw.sample(r, a, b, K), tt, nn, lam, S, w,
            c0, K=400, n_sub=nsub, alpha=0.1)
        return out["CVaR"]

    before = twin_cvar(t, n0, 99)
    t_r, n_r, hist = ft.robustify(lam, t, n0, S, w, c0, tw, box,
                                  alpha=0.1, K=48, steps=40, lr=4e-3,
                                  seed=0, n_sub=nsub)
    after = twin_cvar(t_r, n_r, 99)
    assert after > before
