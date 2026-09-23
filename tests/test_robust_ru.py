"""CVaR without the finite-K bias, twin ensembles, and robust design
through the general model path (0.7.0)."""
import numpy as np
import pytest

import fabtwin as ft
from fabtwin.merits import LinearMerit, OpticalModel
from fabtwin.robust import (TwinEnsemble, _draws, cvar_objective_and_grad,
                            ru_objective_and_grad)

LAM = np.linspace(0.45, 0.65, 41)
S = ft.dispersion_shape(LAM, 0.55)
W, C = ft.notch_weights(LAM, 0.532, 0.012, 0.02)


@pytest.fixture(scope="module")
def setup():
    N = 4
    box = ft.DesignBox(N, 0.03, 0.15, 1.7, 2.3)
    rng = np.random.default_rng(0)
    t = rng.uniform(0.05, 0.12, N)
    n0 = rng.uniform(1.8, 2.2, N)
    rt, rn, ftt, fn = ft.PAPER_PROCESS.trace_dataset(
        np.tile(t, (20, 1)), np.tile(n0, (20, 1)), 10, rng)
    x = ft.errors_from_traces(rt, rn, ftt, fn)
    return box, t, n0, x, ft.GaussianTwin(x)


def test_ru_value_and_gradient_are_exact_for_a_fixed_batch(setup):
    box, t, n0, x, twin = setup
    z = np.random.default_rng(1).normal(size=(50, twin.dim))
    tau = 0.55
    v, gt, gn, gtau, J = ru_objective_and_grad(
        LAM, t, n0, tau, S, W, C, twin, z, 0.1, box, n_sub=1.46)
    assert v == pytest.approx(tau - np.maximum(tau - J, 0).sum()
                              / (0.1 * 50), abs=1e-14)
    assert gtau == pytest.approx(1 - np.mean(J < tau) / 0.1, abs=1e-14)
    h = 1e-6
    for i in range(t.size):
        for arr, g in ((t, gt), (n0, gn)):
            a1, a2 = arr.copy(), arr.copy()
            a1[i] += h
            a2[i] -= h
            args1 = (a1, n0) if arr is t else (t, a1)
            args2 = (a2, n0) if arr is t else (t, a2)
            v1 = ru_objective_and_grad(LAM, *args1, tau, S, W, C, twin, z,
                                       0.1, box, n_sub=1.46)[0]
            v2 = ru_objective_and_grad(LAM, *args2, tau, S, W, C, twin, z,
                                       0.1, box, n_sub=1.46)[0]
            assert (v1 - v2) / (2 * h) == pytest.approx(g[i], rel=1e-5,
                                                        abs=1e-8)


def test_maximizing_ru_over_tau_gives_the_cvar(setup):
    box, t, n0, x, twin = setup
    z = np.random.default_rng(2).normal(size=(200, twin.dim))
    J = _draws(LAM, t, n0, S, W, C, twin, z, box, 1.0, 1.46, None)[0]
    # alpha K = 20 is an integer: the maximum over tau is attained at
    # the 20th smallest value and equals the mean of the 20 smallest
    tau = np.sort(J)[19]
    v = ru_objective_and_grad(LAM, t, n0, tau, S, W, C, twin, z, 0.1, box,
                              n_sub=1.46)[0]
    assert v == pytest.approx(ft.cvar(J, 0.1), abs=1e-13)
    for d in (-0.01, 0.01):
        assert ru_objective_and_grad(LAM, t, n0, tau + d, S, W, C, twin, z,
                                     0.1, box, n_sub=1.46)[0] <= v + 1e-15


def test_sort_estimator_is_biased_and_ru_is_not(setup):
    """Average many K = 20 minibatch gradients drawn from one large pool
    of draws, and compare with the pool's own CVaR gradient."""
    box, t, n0, x, twin = setup
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(6000, twin.dim))
    J, gT, gN = _draws(LAM, t, n0, S, W, C, twin, Z, box, 1.0, 1.46, None)
    alpha, K, B = 0.1, 20, 20000
    tau = np.quantile(J, alpha)
    G = np.hstack([gT, gN])
    ref = ((J < tau) @ G) / (alpha * J.size)
    gs = np.empty((B, G.shape[1]))
    gr = np.empty_like(gs)
    q = int(np.ceil(alpha * K))
    for b in range(B):
        idx = rng.integers(0, J.size, K)
        Jb, Gb = J[idx], G[idx]
        gs[b] = Gb[np.argsort(Jb)[:q]].mean(0)
        gr[b] = ((Jb < tau) / (alpha * K)) @ Gb
    z_sort = (gs.mean(0) - ref) / (gs.std(0, ddof=1) / np.sqrt(B))
    z_ru = (gr.mean(0) - ref) / (gr.std(0, ddof=1) / np.sqrt(B))
    assert np.abs(z_ru).max() < 4.0
    assert np.abs(z_sort).max() > 6.0


def test_robustify_ru_and_ensemble_raise_the_fabricated_cvar():
    N = 6
    box = ft.DesignBox(N, 0.03, 0.15, 1.7, 2.3)
    t, n0, _, _ = ft.inverse_design(LAM, S, W, C, box, n_probe=60,
                                    n_seed=2, n_iter=25, n_sub=1.46)
    rng = np.random.default_rng(1)
    rt_, rn_ = ft.design_recipes(box, 10)
    tr = ft.PAPER_PROCESS.trace_dataset(rt_, rn_, 10, rng)
    x = ft.errors_from_traces(*tr)

    def score(tt, nn):
        samp = lambda a, b, K: ft.PAPER_PROCESS.ensemble(
            a, b, K, np.random.default_rng(99))
        return ft.evaluate_under_process(samp, tt, nn, LAM, S, W, C, 400,
                                         n_sub=1.46, alpha=0.1)["CVaR"]
    base = score(t, n0)
    for twin in (ft.GaussianTwin(x), TwinEnsemble(x, n_members=8)):
        a, b, hist = ft.robustify(LAM, t, n0, S, W, C, twin, box,
                                  alpha=0.1, K=48, steps=60, n_sub=1.46,
                                  estimator="ru")
        assert score(a, b) > base
        assert np.all(a >= box.t_lo) and np.all(a <= box.t_hi)


def test_default_robustify_is_unchanged_and_model_path_agrees(setup):
    box, t, n0, x, twin = setup
    z = np.random.default_rng(4).normal(size=(30, twin.dim))
    v0, gt0, gn0, J0 = cvar_objective_and_grad(LAM, t, n0, S, W, C, twin,
                                               z, 0.1, box, n_sub=1.46)
    model = OpticalModel(LinearMerit(W, C, "T"))
    v1, gt1, gn1, J1 = cvar_objective_and_grad(
        LAM, t, n0, S, None, None, twin, z, 0.1, box, n_sub=1.46,
        model=model)
    assert v0 == pytest.approx(v1, abs=1e-13)
    assert np.allclose(gt0, gt1, rtol=1e-10, atol=1e-12)
    assert np.allclose(gn0, gn1, rtol=1e-10, atol=1e-12)


def test_twin_ensemble(setup):
    box, t, n0, x, twin = setup
    ens = TwinEnsemble(x, n_members=5, seed=3)
    z = np.random.default_rng(5).normal(size=(10, x.shape[1]))
    xe = ens.transform(z)
    for k in range(10):
        m = ens.members[k % 5]
        assert np.allclose(xe[k], m.transform(z[k:k + 1])[0])
    means = np.array([m.mu for m in ens.members])
    assert np.any(np.ptp(means, axis=0) > 0)       # members differ
    with pytest.raises(ValueError):
        TwinEnsemble(x[:3])
    with pytest.raises(ValueError):
        TwinEnsemble(x, n_members=1)
    with pytest.raises(ValueError):
        ft.robustify(LAM, t, n0, S, W, C, twin, box, estimator="median")
    with pytest.raises(ValueError):
        ft.robustify(LAM, t, n0, S, W, C, twin, box, estimator="ru",
                     mean_variance=True)
