"""In-run correction of the remaining layers, conditional twins, and the
refined calibration design (0.7.0)."""
import numpy as np
import pytest

import fabtwin as ft
from fabtwin.correct import reoptimize_remaining
from fabtwin.lab import maximin_distance

LAM = np.linspace(0.45, 0.65, 61)
S = ft.dispersion_shape(LAM, 0.55)
W, C = ft.notch_weights(LAM, 0.532, 0.012, 0.02)


def _merit(t, n):
    return float(ft.merit(ft.transmittance(LAM, t, n[:, None] * S, 1.0,
                                           1.46), W, C))


@pytest.fixture(scope="module")
def design():
    N = 8
    box = ft.DesignBox(N, 0.03, 0.15, 1.7, 2.3)
    t, n, _, _ = ft.inverse_design(LAM, S, W, C, box, n_probe=80, n_seed=2,
                                   n_iter=30, n_sub=1.46)
    rng = np.random.default_rng(3)
    rt, rn = ft.design_recipes(box, 10)
    x = ft.errors_from_traces(*ft.PAPER_PROCESS.trace_dataset(rt, rn, 12,
                                                              rng))
    return box, t, n, ft.GaussianTwin(x)


def test_conditional_twin_is_the_exact_gaussian_conditional(design):
    box, t, n, twin = design
    obs = np.array([0, 1, 8])
    val = np.array([0.03, -0.01, 0.02])
    c = twin.conditional(obs, val)
    un = np.setdiff1d(np.arange(twin.dim), obs)
    Sig = twin.cov
    K = Sig[np.ix_(un, obs)] @ np.linalg.inv(Sig[np.ix_(obs, obs)])
    assert np.allclose(c.mu[un], twin.mu[un] + K @ (val - twin.mu[obs]),
                       atol=1e-12)
    assert np.allclose(c.cov[np.ix_(un, un)],
                       Sig[np.ix_(un, un)] - K @ Sig[np.ix_(obs, un)],
                       atol=1e-14)
    z = np.random.default_rng(0).normal(size=(2000, twin.dim))
    xs = c.transform(z)
    assert np.all(xs[:, obs] == val)             # measured errors stay fixed
    # the sample covariance of the draws matches the conditional one
    assert np.abs(np.cov(xs[:, un].T) - c.cov[np.ix_(un, un)]).max() \
        < 0.15 * np.abs(c.cov[np.ix_(un, un)]).max()
    # Monte Carlo: regression of x_u on x_o over joint draws gives K
    zj = np.random.default_rng(1).normal(size=(200000, twin.dim))
    xj = twin.transform(zj)
    coef = np.linalg.lstsq(np.c_[np.ones(len(xj)), xj[:, obs]], xj[:, un],
                           rcond=None)[0][1:].T
    assert np.abs(coef - K).max() < 0.02 * max(1.0, np.abs(K).max())
    with pytest.raises(ValueError):
        twin.conditional([0, 0], [0.1, 0.1])


def _done(first, N, m):
    return np.arange(m) if first == "incidence" else np.arange(N - m, N)


@pytest.mark.parametrize("first", ["substrate", "incidence"])
def test_reoptimization_never_worse_and_keeps_deposited_layers(design,
                                                               first):
    box, t, n, twin = design
    rng = np.random.default_rng(4)
    m = 4
    d = _done(first, t.size, m)
    for _ in range(5):
        tf, nf = ft.PAPER_PROCESS.corrupt(t, n, rng)
        out = reoptimize_remaining(LAM, t, n, S, W, C, m, tf[d], nf[d],
                                   box, n_sub=1.46, n_iter=25, first=first)
        assert out["J_after"] >= out["J_before"] - 1e-12
        assert np.array_equal(out["t"][d], tf[d])
        assert np.array_equal(out["n"][d], nf[d])
        t0, n0 = t.copy(), n.copy()
        t0[d], n0[d] = tf[d], nf[d]
        assert out["J_before"] == pytest.approx(_merit(t0, n0), abs=1e-12)


@pytest.mark.parametrize("first", ["substrate", "incidence"])
def test_correction_helps_on_the_reference_process(design, first):
    box, t, n, twin = design
    rng = np.random.default_rng(5)
    m = 4
    d = _done(first, t.size, m)
    rest = np.setdiff1d(np.arange(t.size), d)
    gain_det, gain_rob = [], []
    for run in range(12):
        tf, nf = ft.PAPER_PROCESS.corrupt(t, n, rng)
        xt, xn = tf / t - 1, nf - n
        base = _merit(tf, nf)

        def finish(out):
            # the remaining layers suffer the SAME errors as uncorrected
            t2, n2 = out["t"].copy(), out["n"].copy()
            t2[rest] *= 1 + xt[rest]
            n2[rest] += xn[rest]
            return _merit(t2, n2) - base
        gain_det.append(finish(reoptimize_remaining(
            LAM, t, n, S, W, C, m, tf[d], nf[d], box, n_sub=1.46,
            n_iter=25, first=first)))
        if run < 4:
            gain_rob.append(finish(reoptimize_remaining(
                LAM, t, n, S, W, C, m, tf[d], nf[d], box, twin=twin,
                n_sub=1.46, n_iter=25, steps=30, K=32, first=first)))
    assert np.mean(gain_det) > 0
    assert np.mean(gain_rob) > 0


def test_robust_correction_reproduces_the_measured_stack(design):
    # the robust branch keeps the deposited layers at the recipe and
    # fixes their errors at the measured ones; recipe (1 + x) must be
    # the measured stack
    box, t, n, twin = design
    m = 3
    tf, nf = ft.PAPER_PROCESS.corrupt(t, n, np.random.default_rng(6))
    d = np.arange(t.size - m, t.size)
    obs = np.concatenate([d, t.size + d])
    x_obs = np.concatenate([tf[d] / t[d] - 1, nf[d] - n[d]])
    c = twin.conditional(obs, x_obs)
    tt, nn = ft.apply_errors(t, n, c.transform(
        np.random.default_rng(7).normal(size=(5, twin.dim))))
    assert np.abs(tt[:, d] - tf[d]).max() < 1e-15
    assert np.abs(nn[:, d] - nf[d]).max() < 1e-15
    ens = ft.TwinEnsemble(ft.errors_from_traces(*ft.PAPER_PROCESS
                          .trace_dataset(np.tile(t, (10, 1)),
                                         np.tile(n, (10, 1)), 4,
                                         np.random.default_rng(8))),
                          n_members=4)
    out = reoptimize_remaining(LAM, t, n, S, W, C, m, tf[d], nf[d], box,
                               twin=ens, n_sub=1.46, n_iter=10, steps=5,
                               K=8)
    assert np.array_equal(out["t"][d], tf[d])


def test_reoptimization_refusals(design):
    box, t, n, twin = design
    with pytest.raises(ValueError):
        reoptimize_remaining(LAM, t, n, S, W, C, 0, [], [], box)
    with pytest.raises(ValueError):
        reoptimize_remaining(LAM, t, n, S, W, C, 3, t[:2], n[:2], box)
    with pytest.raises(ValueError):
        reoptimize_remaining(LAM, t, n, S, W, C, 2, t[:2], n[:2], box,
                             first="middle")


def test_refined_calibration_design_is_never_worse():
    for N, k in ((3, 6), (4, 12), (8, 20)):
        box = ft.DesignBox(N, 0.03, 0.15, 1.7, 2.3)
        g = maximin_distance(box, *ft.design_recipes(box, k))
        r = maximin_distance(box, *ft.design_recipes(box, k, refine=True))
        assert r >= g - 1e-12
    box = ft.DesignBox(3, 0.03, 0.15, 1.7, 2.3)
    assert maximin_distance(box, *ft.design_recipes(box, 6, refine=True)) \
        > 1.2 * maximin_distance(box, *ft.design_recipes(box, 6))
    # the default (no refine) is unchanged from the greedy rule
    a = ft.design_recipes(box, 6)
    b = ft.design_recipes(box, 6, refine=False)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
