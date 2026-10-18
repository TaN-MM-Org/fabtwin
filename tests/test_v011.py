"""Tests of 0.11.0: band edges that fall on the wavelength grid, ties in
the drift test, the batched forward solver, and the whole-run
monitoring tools `monitoring_errors` and `simulate_monitored_runs`.

Independent references: integer grid arithmetic for the band edges;
the identity of the random splits of the drift test (integer set
comparison, no floating point); the per-design loops of 0.10.1, kept
here as references; the closed form of one layer on a substrate,
inverted by hand; root finding on `stack_rt` (scipy's brentq) for the
simulated stops; central finite differences of the simulator for the
propagation matrix; and the scatter of 2000 simulated runs.
"""
import itertools

import numpy as np
import pytest
from scipy.optimize import brentq

import fabtwin as ft
from fabtwin import design, tmm
from fabtwin.fidelity import energy_distance

# ------------------------- band edges on the grid -------------------------


def _masks(w):
    return w > 0, w < 0               # pass, stop (weights 0.5/n, -0.5/n)


def test_band_edges_on_the_grid_belong_to_their_band():
    """A grid point exactly on a band edge (in decimal) belongs to the
    band. 0.5325 + 0.0125 is 0.5449999999999999 in binary, so 0.10.x
    dropped the grid point 0.545 from the stop band [0.52, 0.545] (5
    points instead of 6), and 0.55 + 0.02 + 0.01 is 0.5800000000000001,
    so 0.58 was dropped from the upper pass band while 0.52 was kept in
    the lower one. Reference: the bands counted on the integer grid
    index i (lam_i = lam_0 + i * step)."""
    lam = np.linspace(0.45, 0.70, 51)                 # 5 nm, i = 0..50
    i = np.arange(51)
    w, c = ft.notch_weights(lam, 0.5325, 0.0125, 0.02)
    pas, stop = _masks(w)
    np.testing.assert_array_equal(stop, (i >= 14) & (i <= 19))   # 0.52..0.545
    np.testing.assert_array_equal(pas, (i <= 10) | (i >= 23))    # 0.50, 0.565
    assert stop.sum() == 6 and c == 0.5
    np.testing.assert_allclose(w[stop], -0.5 / 6, rtol=0, atol=1e-15)

    lam = np.linspace(0.40, 0.70, 61)                 # 5 nm, i = 0..60
    i = np.arange(61)
    pas, stop = _masks(ft.notch_weights(lam, 0.55, 0.02, 0.01)[0])
    np.testing.assert_array_equal(stop, (i >= 26) & (i <= 34))   # 0.53..0.57
    np.testing.assert_array_equal(pas, (i <= 24) | (i >= 36))    # 0.52, 0.58
    assert pas[:30].sum() == 25 and pas[31:].sum() == 25          # symmetric

    pas, stop = _masks(ft.bandpass_weights(lam, 0.55, 0.60, 0.02)[0])
    # in the bandpass merit the pass band is "inside" (weight > 0)
    np.testing.assert_array_equal(pas, (i >= 30) & (i <= 40))
    np.testing.assert_array_equal(stop, (i <= 26) | (i >= 44))


def test_band_edges_off_the_grid_are_unchanged():
    """Where no grid point lies within 1e-6 um of an edge, the weights
    are those of the 0.10.x formula exactly (the tolerance is 1e-9 of
    the largest wavelength)."""
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(200):
        lam = np.sort(rng.uniform(0.4, 0.9, rng.integers(20, 200)))
        c, h, g = rng.uniform(0.5, 0.8), rng.uniform(0.005, 0.05), \
            rng.uniform(0.0, 0.05)
        edges = np.array([c - h, c + h, c - h - g, c + h + g])
        if np.min(np.abs(lam[:, None] - edges[None, :])) < 1e-6:
            continue
        stop = (lam >= c - h) & (lam <= c + h)
        pas = (lam <= c - h - g) | (lam >= c + h + g)
        if not stop.any() or not pas.any():
            continue
        w_old = 0.5 * pas / pas.sum() - 0.5 * stop / stop.sum()
        np.testing.assert_array_equal(ft.notch_weights(lam, c, h, g)[0],
                                      w_old)
        checked += 1
    assert checked > 100

# ------------------------- drift test: ties -------------------------


def _split_reference(n_old, n_new, n_perm, seed):
    """(1 + number of permutations that reproduce the observed split,
    or swap the two groups when they have equal sizes) / (1 + n_perm),
    from the identity of the index sets alone (no floating point). The
    permutations are drawn exactly as `drift_test` draws them."""
    rng = np.random.default_rng(seed)
    n = n_old + n_new
    first = set(range(n_old))
    rest = set(range(n_old, n))
    count = 0
    for _ in range(n_perm):
        s = set(rng.permutation(n)[:n_old].tolist())
        if s == first or (n_old == n_new and s == rest):
            count += 1
    return (1.0 + count) / (1.0 + n_perm)


@pytest.mark.parametrize("data_seed,k,m", [(1, 2, 2), (10, 2, 2), (0, 3, 3),
                                           (6, 2, 3), (3, 2, 2)])
def test_drift_test_counts_ties_in_small_groups(data_seed, k, m):
    """Two groups far apart: only the observed split (and, for equal
    sizes, its swap) reaches the observed energy distance, every other
    split is clearly smaller (checked by enumerating all of them). So
    the p-value must be (1 + number of permutations giving those
    splits) / (1 + n_perm). Recomputed from reordered rows, the
    statistic of the same split can come out a few units of the last
    digit below the observed one, and 0.10.x then did not count it.
    Before -> after: data seed 1, 2 + 2 runs, p = 0.036 -> 0.315 (with 4
    runs no p-value below 1/3 is possible); seed 10, 2 + 2: 0.065 ->
    0.315; seed 0, 3 + 3: 0.082 -> 0.103; seed 6, 2 + 3: 0.052 -> 0.106;
    seed 3, 2 + 2 was right before (0.315) and stays so."""
    rng = np.random.default_rng(data_seed)
    a = rng.normal(size=(k, 3))
    b = rng.normal(size=(m, 3)) + 6.0
    pool = np.vstack([a, b])
    pool = (pool - pool.mean(0)) / pool.std(axis=0, ddof=1)
    obs = energy_distance(pool[:k], pool[k:])
    for s in itertools.combinations(range(k + m), k):
        s = set(s)
        if s == set(range(k)) or (k == m and s == set(range(k, k + m))):
            continue
        rest = sorted(set(range(k + m)) - s)
        assert energy_distance(pool[sorted(s)], pool[rest]) < 0.9 * obs
    p = ft.drift_test(a, b, n_perm=999, seed=0)["p_value"]
    assert p == _split_reference(k, m, 999, 0)
    if k + m == 4:
        assert p > 0.25


def test_drift_test_small_groups_never_go_below_the_exact_minimum():
    """Under the null (both groups from one distribution) with 3 + 3
    runs there are 20 splits, and the observed one ties with its swap,
    so the exact permutation p-value is at least 2/20. With 999 random
    permutations the estimate of a p-value of 0.1 has a standard error
    of 0.0095; none of 60 seeded data sets may come out below 0.06
    (more than 4 standard errors under 0.1). 0.10.x gave 0.046 for one
    of them (smallest now: 0.087)."""
    rng = np.random.default_rng(7)
    ps = [ft.drift_test(rng.normal(size=(3, 2)), rng.normal(size=(3, 2)),
                        n_perm=999, seed=s)["p_value"] for s in range(60)]
    assert min(ps) >= 0.06

# ------------------- batched forward solver (speed) -------------------


def _loop_T(lam, t_k, n_k, S, n_inc, n_sub):
    """The per-design loop of 0.10.1 (reference)."""
    S2 = S if S.ndim == 2 else S[None, :]
    return np.array([tmm.transmittance(lam, t_k[k], n_k[k][:, None] * S2,
                                       n_inc, n_sub)
                     for k in range(t_k.shape[0])])


def _close(a, b):
    """Equal to 1e-12 of the largest value (not bit for bit: vectorized
    math kernels may round the last digit differently on other
    processors; see 0.10.1)."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    scale = max(np.max(np.abs(b)), 1e-300)
    assert np.max(np.abs(a - b)) <= 1e-12 * scale


def test_batched_forward_equals_the_per_design_loop(monkeypatch):
    """`tmm._transmittance_batch` against one `transmittance` call per
    design, on 30 random cases: 1 to 20 layers, shared and per-layer
    dispersion, scalar and dispersive substrates, absorbing layers, and
    small chunks (several chunks per call)."""
    rng = np.random.default_rng(11)
    lam = np.linspace(0.42, 0.78, 37)
    for case in range(30):
        if case % 3 == 0:
            monkeypatch.setattr(tmm, "_BATCH_COLUMNS", 97)
        else:
            monkeypatch.setattr(tmm, "_BATCH_COLUMNS", 2 ** 15)
        N = int(rng.integers(1, 21))
        K = int(rng.integers(1, 120))
        t = rng.uniform(0.01, 0.3, (K, N))
        n = rng.uniform(1.3, 2.5, (K, N))
        if case % 5 == 4:
            n = n + 1j * rng.uniform(0.0, 0.2, (K, N))
        S = ft.dispersion_shape(lam, 0.55)
        if case % 2:
            S = np.vstack([S] * N) * rng.uniform(0.95, 1.05, (N, 1))
        ns = 1.52 if case % 4 == 0 else ft.SIO2_MALITSON1965.n(lam)
        _close(tmm._transmittance_batch(lam, t, n, S, 1.0, ns),
               _loop_T(lam, t, n, S, 1.0, ns))


def _setup():
    lam = np.linspace(0.45, 0.65, 41)
    S = ft.dispersion_shape(lam, 0.550)
    nsub = ft.SIO2_MALITSON1965.n(lam)
    w, c0 = ft.notch_weights(lam, 0.532, 0.015, 0.030)
    box = ft.DesignBox(8, 0.020, 0.120, 1.6, 2.4)
    return lam, S, nsub, w, c0, box


def test_scoring_and_probing_equal_the_0_10_loops():
    """`evaluate_under_process`, `induced_merits`, `random_search` and
    the probes of `inverse_design` now score all designs in one solver
    call; the 0.10.1 per-design loops are the reference."""
    lam, S, nsub, w, c0, box = _setup()
    rng = np.random.default_rng(2)
    t = rng.uniform(0.03, 0.11, 8)
    n0 = rng.uniform(1.7, 2.3, 8)

    def sampler(tt, nn, K):
        return ft.PAPER_PROCESS.ensemble(tt, nn, K, np.random.default_rng(5))

    out = ft.evaluate_under_process(sampler, t, n0, lam, S, w, c0, 300,
                                    n_sub=nsub, alpha=0.1)
    D, Nn = sampler(t, n0, 300)
    ref = ft.merit(_loop_T(lam, D, Nn, S, 1.0, nsub), w, c0)
    _close(out["samples"], ref)
    assert abs(out["CVaR"] - ft.cvar(ref, 0.1)) <= 1e-12

    x = rng.normal(0.0, 0.02, (200, 16))
    tt, nn = ft.apply_errors(t, n0, x)
    _close(ft.induced_merits(x, t, n0, lam, S, w, c0, n_sub=nsub),
           ft.merit(_loop_T(lam, tt, nn, S, 1.0, nsub), w, c0))

    tb, nb, Jb, hist = ft.random_search(lam, S, w, c0, box, 150,
                                        n_sub=nsub, seed=4)
    tr, nr = box.sample(np.random.default_rng(4), 150)
    Jr = ft.merit(_loop_T(lam, tr, nr, S, 1.0, nsub), w, c0)
    _close(hist, np.maximum.accumulate(Jr))
    assert int(np.argmax(Jr)) == int(np.argmax(hist == hist[-1]))
    _close(tb, tr[np.argmax(Jr)])

    # inverse_design: the probe merits, and therefore the seeds and the
    # whole result, against the 0.10.1 loop
    td, nd, Jd, traj = ft.inverse_design(lam, S, w, c0, box, n_probe=40,
                                         n_seed=2, n_iter=15, n_sub=nsub,
                                         seed=0)
    tp, npr = box.sample(np.random.default_rng(0), 40)
    Jp = np.array([design._merit_of(lam, tp[r], npr[r], S, w, c0, 1.0,
                                    nsub) for r in range(40)])
    _close(traj[:40], np.maximum.accumulate(Jp))
    best = (None, None, -np.inf)
    for sidx in np.argsort(Jp)[::-1][:2]:
        r = ft.adam_ascent(lam, tp[sidx], npr[sidx], S, w, c0, box,
                           n_iter=15, n_sub=nsub)
        if r[2] > best[2]:
            best = r[:3]
    _close(td, best[0])
    _close(nd, best[1])
    assert abs(Jd - best[2]) <= 1e-12

# ------------------- whole-run monitoring (new) -------------------


def _one_layer_stop(T, n0, n1, ns, lam, t_recipe):
    """Closed form: the thickness at which one layer of index n1 on a
    substrate ns (light from n0, normal incidence) has transmittance T,
    on the branch of T(delta) that contains the recipe thickness.
    T = 4 n0 ns / [a cos^2 d + b sin^2 d] with a = (n0 + ns)^2,
    b = (n0 ns / n1 + n1)^2, so sin^2 d = (4 n0 ns / T - a) / (b - a)."""
    a = (n0 + ns) ** 2
    b = (n0 * ns / n1 + n1) ** 2
    s2 = (4 * n0 * ns / T - a) / (b - a)
    base = np.arcsin(np.sqrt(s2))                       # in [0, pi/2]
    d_r = 2 * np.pi * n1 * t_recipe / lam
    m = int(np.floor(d_r / (np.pi / 2)))                # quarter index
    d = m * np.pi / 2 + (base if m % 2 == 0 else np.pi / 2 - base)
    return d * lam / (2 * np.pi * n1)


def _one_layer_slope(n0, n1, ns, lam, t):
    a = (n0 + ns) ** 2
    b = (n0 * ns / n1 + n1) ** 2
    d = 2 * np.pi * n1 * t / lam
    den = a * np.cos(d) ** 2 + b * np.sin(d) ** 2
    return -4 * n0 * ns * (b - a) * np.sin(2 * d) / den ** 2 \
        * 2 * np.pi * n1 / lam


def test_one_layer_matches_the_closed_form():
    """One layer: the simulated stop for given signal errors equals the
    hand-inverted closed form, and the predicted error is
    sigma / |dT/dt| with the closed-form slope."""
    lam, n1, ns, t1 = 0.55, 2.1, 1.52, 0.095
    e = np.array([[0.0], [3e-3], [-2e-3], [1e-4]])
    T_cut = ft.transmittance([lam], [t1], [n1], 1.0, ns)[0]
    got = ft.simulate_monitored_runs(lam, [t1], [n1], noise=e, n_sub=ns)
    want = [_one_layer_stop(T_cut - ek, 1.0, n1, ns, lam, t1)
            for ek in e[:, 0]]
    np.testing.assert_allclose(got[:, 0], want, rtol=0, atol=1e-13)
    assert got[0, 0] == pytest.approx(t1, abs=1e-14)
    me = ft.monitoring_errors(lam, [t1], [n1], 1e-3, n_sub=ns)
    slope = _one_layer_slope(1.0, n1, ns, lam, t1)
    assert me.sensitivity[0, 0] == pytest.approx(slope, rel=1e-12)
    assert me.sigma_t[0] == pytest.approx(1e-3 / abs(slope), rel=1e-12)
    assert me.propagation[0, 0] == pytest.approx(-1 / slope, rel=1e-12)


T4 = np.array([0.075, 0.098, 0.066, 0.118])
N4 = np.array([2.1, 1.46, 2.1, 1.46])


def _brentq_run(lam, t, n, e, ns, off=0.0):
    """One run simulated independently: layer by layer, scipy's brentq
    on `stack_rt` in a bracket of +/- 20 % around the recipe (no turning
    point inside for this recipe and wavelength)."""
    N = t.size
    dep = np.empty(N)
    for j in range(N - 1, -1, -1):
        cut = ft.stack_rt([lam], t[j:], n[j:], 1.0, ns)[1][0]

        def f(tau):
            return ft.stack_rt([lam], np.concatenate([[tau], dep[j + 1:]]),
                               n[j:], 1.0, ns)[1][0] - (cut - off - e[j])
        dep[j] = brentq(f, 0.8 * t[j], 1.2 * t[j], xtol=1e-15, rtol=1e-15)
    return dep


def test_simulated_runs_equal_independent_root_finding():
    """`simulate_monitored_runs` (vectorized bisection on the gradient
    module's forward values) against brentq on `stack_rt`, run by run,
    with errors carried from layer to layer, to 1e-12 um."""
    lam = 0.47
    e = np.random.default_rng(8).normal(0.0, 1e-3, (6, 4))
    got = ft.simulate_monitored_runs(lam, T4, N4, noise=e, n_sub=1.52,
                                     signal_offset=5e-4)
    for r in range(6):
        np.testing.assert_allclose(
            got[r], _brentq_run(lam, T4, N4, e[r], 1.52, 5e-4),
            rtol=0, atol=1e-12)
    zero = ft.simulate_monitored_runs(lam, T4, N4, noise=np.zeros((2, 4)),
                                      n_sub=1.52)
    np.testing.assert_allclose(zero, np.tile(T4, (2, 1)), rtol=0,
                               atol=1e-14)


def test_propagation_matrix_is_the_derivative_of_the_simulated_run():
    """G = -A^-1 against central finite differences of the simulated
    run with respect to the signal error at each stop (step 1e-6; the
    truncation error is O(step^2 |d2 t / d e2|), well below the 1e-6
    relative tolerance), and A is upper triangular (a stop depends only
    on the layers deposited before it)."""
    lam = 0.47
    me = ft.monitoring_errors(lam, T4, N4, 1e-3, n_sub=1.52)
    assert np.all(np.tril(me.sensitivity, -1) == 0.0)
    h = 1e-6
    E = h * np.eye(4)
    rp = ft.simulate_monitored_runs(lam, T4, N4, noise=E, n_sub=1.52)
    rm = ft.simulate_monitored_runs(lam, T4, N4, noise=-E, n_sub=1.52)
    fd = (rp - rm).T / (2 * h)              # column k: d t / d e_k
    G = me.propagation
    assert np.max(np.abs(fd - G)) <= 1e-6 * np.max(np.abs(G))
    # the independent (per-layer) errors are those of monitoring_plan
    p = ft.monitoring_plan(lam, T4, N4, n_sub=1.52, sigma_signal=1e-3)
    np.testing.assert_allclose(me.sigma_t_independent, p.sigma_t,
                               rtol=1e-12)
    # the first layer deposited has nothing below it: no propagation
    assert me.sigma_t[3] == pytest.approx(me.sigma_t_independent[3],
                                          rel=1e-12)
    # an offset moves every layer by G applied to it (second order is
    # O(offset^2): 1e-5 gives a relative difference near 1e-5)
    me_o = ft.monitoring_errors(lam, T4, N4, 1e-3, n_sub=1.52,
                                signal_offset=1e-5)
    r_o = ft.simulate_monitored_runs(lam, T4, N4, noise=np.zeros((1, 4)),
                                     n_sub=1.52, signal_offset=1e-5)
    np.testing.assert_allclose(r_o[0] - T4, me_o.mean_t, rtol=1e-3)
    np.testing.assert_allclose(me_o.mean_t, G @ np.full(4, 1e-5),
                               rtol=1e-12)


def test_linear_covariance_matches_2000_simulated_runs():
    """The predicted standard deviations and correlations against 2000
    simulated runs (0.05 % signal noise, nonlinearity below 0.02). The
    standard error of a sample standard deviation is sd / sqrt(2 (K -
    1)), 1.6 % here, and of a correlation (1 - rho^2) / sqrt(K), at most
    0.023; the test allows 6 % and 0.08 (about 4 standard errors). The
    propagated errors are clearly not the independent ones: layer 0's is
    2.9 times larger at this wavelength."""
    lam, sig = 0.47, 5e-4
    me = ft.monitoring_errors(lam, T4, N4, sig, n_sub=1.52)
    assert np.all(me.nonlinearity < 0.02)
    runs = ft.simulate_monitored_runs(lam, T4, N4, sig, 2000, seed=21,
                                      n_sub=1.52)
    assert np.all(np.isfinite(runs))
    dt = runs - T4
    sd = dt.std(axis=0, ddof=1)
    np.testing.assert_allclose(sd / me.sigma_t, 1.0, atol=0.06)
    np.testing.assert_allclose(np.corrcoef(dt.T), me.correlation, atol=0.08)
    assert np.all(np.abs(dt.mean(axis=0)) < 4 * me.sigma_t / np.sqrt(2000))
    assert me.sigma_t[0] > 2.5 * me.sigma_t_independent[0]


def test_twin_of_the_monitored_run():
    """`MonitoringErrors.twin()` has the predicted relative covariance
    and mean exactly (L L^T), zero index errors, and its draws scatter
    like the prediction."""
    me = ft.monitoring_errors(0.47, T4, N4, 1e-3, n_sub=1.52,
                              signal_offset=2e-4)
    tw = me.twin()
    assert tw.dim == 8
    np.testing.assert_allclose(tw.L @ tw.L.T, tw.cov, rtol=0, atol=1e-18)
    np.testing.assert_allclose(tw.cov[:4, :4], me.rel_covariance,
                               rtol=1e-12)
    assert np.all(tw.cov[4:] == 0.0) and np.all(tw.mu[4:] == 0.0)
    np.testing.assert_allclose(tw.mu[:4], me.rel_mean_t, rtol=1e-12)
    x = tw.sample_errors(np.random.default_rng(1), 4000)
    assert np.all(x[:, 4:] == 0.0)
    np.testing.assert_allclose(x[:, :4].std(axis=0, ddof=1),
                               me.rel_sigma_t, rtol=0.06)
    tt, nn = tw.sample(np.random.default_rng(1), T4, N4, 3)
    np.testing.assert_array_equal(nn, np.tile(N4, (3, 1)))


def test_monitoring_errors_refusals_and_missed_stops():
    """Refusals, by name; and a simulated stop that the target cannot
    reach on its branch is NaN with a warning, not a wrong number."""
    with pytest.raises(ValueError, match="sigma_signal"):
        ft.monitoring_errors(0.47, T4, N4, 0.0, n_sub=1.52)
    with pytest.raises(ValueError, match="signal_offset"):
        ft.monitoring_errors(0.47, T4, N4, 1e-3, signal_offset=np.nan)
    with pytest.raises(ValueError, match="recipe_t_um"):
        ft.monitoring_errors(0.47, T4[:3], N4, 1e-3)
    with pytest.raises(ValueError, match="positive wavelength"):
        ft.monitoring_errors(-0.47, T4, N4, 1e-3)
    with pytest.raises(ValueError, match="quantity"):
        ft.monitoring_errors(0.47, T4, N4, 1e-3, quantity="A")
    # README example 21 at 0.52 um: layer 3 ends near a turning point and
    # passes a 4.5 nm error on to layer 2, which then cannot be trusted
    with pytest.raises(ValueError, match=r"layer\(s\) 2 end too close"):
        ft.monitoring_errors(0.52, T4, N4, 1e-3, n_sub=1.52)
    with pytest.raises(ValueError, match="not both"):
        ft.simulate_monitored_runs(0.47, T4, N4, 1e-3, noise=np.zeros((1, 4)))
    with pytest.raises(ValueError, match="noise must be"):
        ft.simulate_monitored_runs(0.47, T4, N4, noise=np.zeros((2, 3)))
    with pytest.raises(ValueError, match="n_runs"):
        ft.simulate_monitored_runs(0.47, T4, N4, 1e-3, 2.5)
    with pytest.raises(ValueError, match="sigma_signal and n_runs"):
        ft.simulate_monitored_runs(0.47, T4, N4, 1e-3)
    # one layer, a signal error larger than the whole swing of its curve
    lam, n1, ns = 0.55, 2.1, 1.52
    t1 = 0.9 * lam / (4 * n1)                   # just before the minimum
    e = np.array([[0.0], [0.2], [-0.2]])
    with pytest.warns(RuntimeWarning, match="2 of 3 simulated runs"):
        r = ft.simulate_monitored_runs(lam, [t1], [n1], noise=e, n_sub=ns)
    assert np.isfinite(r[0, 0]) and np.all(np.isnan(r[1:, 0]))
