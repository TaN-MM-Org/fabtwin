"""Tests of 0.13.0: index errors of the deposited layers in a monitored
run, monitoring through a plate that absorbs, all designs of a model=
scoring loop in one solver call, and five fixes (negative thicknesses
and wavelengths in the optics, negative thicknesses from the spectrum
fits, alpha and K of the robust objectives, conditioning a twin on a
component of zero variance, a rate bias of -1 in the reference
process).

Independent references: the closed form of one layer on a substrate,
inverted by hand, and its derivative with respect to the layer index;
root finding on `stack_rt` (scipy's brentq) for the simulated stops;
central finite differences of the simulator; the scatter of 2000
simulated runs; the incoherent solver `inc_tmm` of the `tmm` package
(S. J. Byrnes, arXiv:1603.02720) for the monitor signal through an
absorbing plate; the per-design loops of 0.12.0, kept here; and the
Gaussian conditional on the thickness components alone.
"""
import warnings

import numpy as np
import pytest
from scipy.optimize import brentq

import fabtwin as ft
from fabtwin import merits, robust
from fabtwin.merits import model_merit_and_grad, model_spectra

T4 = np.array([0.075, 0.098, 0.066, 0.118])     # README examples 21, 22
N4 = np.array([2.1, 1.46, 2.1, 1.46])


def _close(a, b, rel=1e-12):
    """a equals b to rel times the largest |b| (not bit for bit: vectorized
    math kernels may round the last digit differently on other
    processors; see 0.10.1)."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    scale = max(float(np.max(np.abs(b))), 1e-300)
    err = float(np.max(np.abs(a - b), initial=0.0))
    assert err <= rel * scale, (err, scale)
    return err / scale


# ------------------ index errors in a monitored run ------------------


def _one_layer_stop(T, n0, n1, ns, lam, t_recipe):
    """Closed form: the thickness at which one layer of index n1 on a
    substrate ns (light from n0, normal incidence) has transmittance T,
    on the branch of T(delta) that contains the recipe thickness.
    T = 4 n0 ns / [a cos^2 d + b sin^2 d], a = (n0 + ns)^2,
    b = (n0 ns / n1 + n1)^2, so sin^2 d = (4 n0 ns / T - a) / (b - a)."""
    a = (n0 + ns) ** 2
    b = (n0 * ns / n1 + n1) ** 2
    s2 = (4 * n0 * ns / T - a) / (b - a)
    base = np.arcsin(np.sqrt(s2))
    d_r = 2 * np.pi * n1 * t_recipe / lam
    m = int(np.floor(d_r / (np.pi / 2)))
    d = m * np.pi / 2 + (base if m % 2 == 0 else np.pi / 2 - base)
    return d * lam / (2 * np.pi * n1)


def _one_layer_dT(n0, n1, ns, lam, t):
    """dT/dt and dT/dn1 of the closed form above, by hand:
    den = a cos^2 d + b sin^2 d, d = 2 pi n1 t / lam,
    d den = (b - a) sin(2 d) dd + b' sin^2 d dn1,
    b' = 2 (n0 ns / n1 + n1) (1 - n0 ns / n1^2),
    dT = -4 n0 ns d den / den^2."""
    a = (n0 + ns) ** 2
    b = (n0 * ns / n1 + n1) ** 2
    bp = 2 * (n0 * ns / n1 + n1) * (1 - n0 * ns / n1 ** 2)
    d = 2 * np.pi * n1 * t / lam
    den = a * np.cos(d) ** 2 + b * np.sin(d) ** 2
    k = -4 * n0 * ns / den ** 2
    dT_dt = k * (b - a) * np.sin(2 * d) * 2 * np.pi * n1 / lam
    dT_dn = k * ((b - a) * np.sin(2 * d) * 2 * np.pi * t / lam
                 + bp * np.sin(d) ** 2)
    return dT_dt, dT_dn


def test_index_error_of_one_layer_matches_the_closed_form():
    """One layer deposited with index n1 + dn but stopped at the cut-off
    of the recipe (index n1): the simulated stop equals the closed form
    solved for the thickness with index n1 + dn, and the predicted
    sensitivities equal the derivatives of the closed form."""
    lam, n1, ns, t1 = 0.55, 2.1, 1.52, 0.095
    T_cut = ft.transmittance([lam], [t1], [n1], 1.0, ns)[0]
    dn = np.array([[0.0], [0.02], [-0.03], [0.004]])
    e = np.array([[0.0], [1e-3], [0.0], [-2e-3]])
    got = ft.simulate_monitored_runs(lam, [t1], [n1], noise=e, n_sub=ns,
                                     index_errors=dn)
    want = [_one_layer_stop(T_cut - e[r, 0], 1.0, n1 + dn[r, 0], ns, lam,
                            t1) for r in range(4)]
    _close(got[:, 0], want, rel=1e-13)               # measured 7e-16
    me = ft.monitoring_errors(lam, [t1], [n1], 1e-3, n_sub=ns,
                              index_cov=[[1e-4]])
    sl_t, sl_n = _one_layer_dT(1.0, n1, ns, lam, t1)
    assert me.sensitivity[0, 0] == pytest.approx(sl_t, rel=1e-12)
    assert me.index_sensitivity[0, 0] == pytest.approx(sl_n, rel=1e-11)
    #                                                  measured 8e-16
    assert me.index_propagation[0, 0] == pytest.approx(-sl_n / sl_t,
                                                       rel=1e-11)
    # one layer: variance of the stop = (noise^2 + (dT/dn)^2 var_n) / slope^2
    var = (1e-6 + sl_n ** 2 * 1e-4) / sl_t ** 2
    assert me.covariance[0, 0] == pytest.approx(var, rel=1e-12)
    assert me.cross_covariance[0, 0] == pytest.approx(-sl_n / sl_t * 1e-4,
                                                      rel=1e-11)


def _brentq_run(lam, t, n_dep, e, ns, off=0.0, n_exit=None, d=None):
    """One run simulated independently: layer by layer, brentq on
    `stack_rt` in a bracket of +/- 20 % around the recipe (no turning
    point inside for this recipe and wavelength). The cut-offs come from
    the recipe indices N4, the deposited layers have the indices n_dep."""
    N = t.size
    dep = np.empty(N)
    off = np.broadcast_to(off, (N,))
    for j in range(N - 1, -1, -1):
        cut = ft.stack_rt([lam], t[j:], N4[j:], 1.0, ns, n_exit=n_exit,
                          sub_thickness_um=d)[1][0]

        def f(tau):
            return ft.stack_rt([lam], np.concatenate([[tau], dep[j + 1:]]),
                               n_dep[j:], 1.0, ns, n_exit=n_exit,
                               sub_thickness_um=d)[1][0] \
                - (cut - off[j] - e[j])
        dep[j] = brentq(f, 0.8 * t[j], 1.2 * t[j], xtol=1e-15, rtol=1e-15)
    return dep


def test_index_errors_in_simulated_runs_equal_independent_root_finding():
    """Four layers, each run with its own index errors, signal noise and
    a drifting offset: every simulated run equals brentq on `stack_rt`,
    layer by layer, to 1e-12 um. Index errors given as (N,) are the same
    in every run, and zero index errors change nothing (exact)."""
    lam = 0.47
    rng = np.random.default_rng(3)
    e = rng.normal(0.0, 1e-3, (5, 4))
    dn = rng.normal(0.0, 0.01, (5, 4))
    off = 2e-4 * np.array([3.0, 2.0, 1.0, 0.0])
    got = ft.simulate_monitored_runs(lam, T4, N4, noise=e, n_sub=1.52,
                                     signal_offset=off, index_errors=dn)
    worst = 0.0
    for r in range(5):
        ref = _brentq_run(lam, T4, N4 + dn[r], e[r], 1.52, off)
        worst = max(worst, float(np.max(np.abs(got[r] - ref))))
    assert worst <= 1e-12                            # measured 1.3e-15 um
    one = ft.simulate_monitored_runs(lam, T4, N4, noise=e, n_sub=1.52,
                                     index_errors=dn[0])
    many = ft.simulate_monitored_runs(lam, T4, N4, noise=e, n_sub=1.52,
                                      index_errors=np.tile(dn[0], (5, 1)))
    assert np.array_equal(one, many)
    zero = ft.simulate_monitored_runs(lam, T4, N4, noise=e, n_sub=1.52,
                                      index_errors=np.zeros(4))
    plain = ft.simulate_monitored_runs(lam, T4, N4, noise=e, n_sub=1.52)
    assert np.array_equal(zero, plain)


def test_index_propagation_is_the_derivative_of_the_simulated_run():
    """G B against central finite differences of the simulated run with
    respect to the index error of each layer (step 1e-6); B is upper
    triangular; a small mean index error moves the simulated layers by
    G B mu (relative 1e-3, the size of the second-order term); and with
    zero index errors the result is that of 0.12.0, bit for bit."""
    lam = 0.47
    me = ft.monitoring_errors(lam, T4, N4, 1e-3, n_sub=1.52,
                              index_cov=np.zeros((4, 4)))
    B = me.index_sensitivity
    assert np.all(np.tril(B, -1) == 0.0)
    h = 1e-6
    E = h * np.eye(4)
    z = np.zeros((4, 4))
    rp = ft.simulate_monitored_runs(lam, T4, N4, noise=z, n_sub=1.52,
                                    index_errors=E)
    rm = ft.simulate_monitored_runs(lam, T4, N4, noise=z, n_sub=1.52,
                                    index_errors=-E)
    fd = (rp - rm).T / (2 * h)                       # column k: dt / dn_k
    _close(fd, me.index_propagation, rel=1e-6)       # measured 1.3e-10
    _close(me.index_propagation, me.propagation @ B, rel=1e-14)  # measured 0

    mu = 1e-5 * np.array([1.0, -2.0, 0.5, 1.5])
    me_m = ft.monitoring_errors(lam, T4, N4, 1e-3, n_sub=1.52,
                                index_mean=mu)
    run = ft.simulate_monitored_runs(lam, T4, N4, noise=np.zeros((1, 4)),
                                     n_sub=1.52, index_errors=mu)
    _close(run[0] - T4, me_m.mean_t, rel=1e-3)       # measured 2.1e-4
    _close(me_m.mean_t, me.index_propagation @ mu, rel=1e-13)  # measured 3e-17

    plain = ft.monitoring_errors(lam, T4, N4, 1e-3, n_sub=1.52)
    for name in ("mean_t", "covariance", "sigma_t", "propagation",
                 "rel_covariance", "nonlinearity"):
        assert np.array_equal(getattr(me, name), getattr(plain, name))
    assert plain.index_sensitivity is None and plain.cross_covariance is None


def _index_case():
    """An index that drifts down the stack (the mean) with correlated
    scatter around it, and 0.05 % signal noise."""
    mu = np.array([0.002, 0.001, 0.0, -0.001])
    i = np.arange(4)
    cov = 0.002 ** 2 * 0.5 ** np.abs(i[:, None] - i[None, :])
    return mu, cov


def test_index_errors_match_4000_simulated_runs():
    """The predicted mean, standard deviations, correlations and the
    correlation between thickness and index errors, against 4000 runs
    simulated without linearization, each with its own index errors
    drawn from (mu, cov) and its own signal noise. Standard errors: of a
    standard deviation sd / sqrt(2 (K - 1)), 1.1 %; of a correlation at
    most 1 / sqrt(K), 0.016; of a mean sd / sqrt(K). The test allows 6 %,
    0.08 and 4 standard errors. (With 20000 runs, outside the tests, the
    standard deviations agree within 1.8 %: the prediction is first
    order.)"""
    lam, sig = 0.47, 5e-4
    mu, cov = _index_case()
    me = ft.monitoring_errors(lam, T4, N4, sig, n_sub=1.52, index_mean=mu,
                              index_cov=cov)
    assert np.all(me.nonlinearity < 0.05)
    K = 4000
    rng = np.random.default_rng(31)
    dn = mu + rng.normal(size=(K, 4)) @ np.linalg.cholesky(cov).T
    e = rng.normal(0.0, sig, (K, 4))
    runs = ft.simulate_monitored_runs(lam, T4, N4, noise=e, n_sub=1.52,
                                      index_errors=dn)
    assert np.all(np.isfinite(runs))
    dt = runs - T4
    sd = dt.std(axis=0, ddof=1)
    assert np.max(np.abs(sd / me.sigma_t - 1)) <= 0.06   # measured 0.016
    assert np.max(np.abs(np.corrcoef(dt.T) - me.correlation)) <= 0.08
    #                                                      measured 0.014
    cross = np.array([[np.corrcoef(dt[:, j], dn[:, k])[0, 1]
                       for k in range(4)] for j in range(4)])
    want = me.cross_covariance / np.outer(me.sigma_t, np.sqrt(np.diag(cov)))
    assert np.max(np.abs(cross - want)) <= 0.08            # measured 0.023
    assert np.all(np.abs(dt.mean(axis=0) - me.mean_t)
                  <= 4 * me.sigma_t / np.sqrt(K))


def test_twin_with_index_errors():
    """`MonitoringErrors.twin()` with index errors: its covariance holds
    the predicted blocks (relative thickness errors, cross terms, index
    errors divided by the dispersion shape at the monitor wavelength),
    L L^T is that covariance, its draws scatter like it, and a layer
    whose index does not vary is no obstacle."""
    lam = 0.47
    mu, cov = _index_case()
    cov[1, :] = cov[:, 1] = 0.0                # layer 1: a fixed index
    me = ft.monitoring_errors(lam, T4, N4, 1e-3, n_sub=1.52, index_mean=mu,
                              index_cov=cov)
    s = np.array([1.02, 1.01, 1.02, 1.01])     # S(lam_mon) per layer
    tw = me.twin(index_shape=s)
    _close(tw.cov[:4, :4], me.rel_covariance, rel=1e-14)  # measured 0
    _close(tw.cov[:4, 4:], me.cross_covariance / np.outer(T4, s),
           rel=1e-14)                              # measured 0
    _close(tw.cov[4:, 4:], cov / np.outer(s, s), rel=1e-14)  # measured 0
    _close(tw.mu, np.concatenate([me.rel_mean_t, mu / s]),
           rel=1e-14)                              # measured 0
    _close(tw.L @ tw.L.T, tw.cov, rel=1e-12)       # measured 2e-18
    x = tw.sample_errors(np.random.default_rng(2), 4000)
    assert np.all(x[:, 5] == mu[1] / s[1])       # a fixed index
    sd = x.std(axis=0, ddof=1)
    keep = np.diag(tw.cov) > 0
    assert np.max(np.abs(sd[keep] / np.sqrt(np.diag(tw.cov))[keep] - 1)) \
        <= 0.06                                      # measured 0.019
    with pytest.raises(ValueError, match="index_shape"):
        me.twin(index_shape=0.0)


def test_index_error_refusals():
    with pytest.raises(ValueError, match="index_mean"):
        ft.monitoring_errors(0.47, T4, N4, 1e-3, index_mean=np.zeros(3))
    with pytest.raises(ValueError, match="index_cov"):
        ft.monitoring_errors(0.47, T4, N4, 1e-3, index_cov=np.eye(3))
    with pytest.raises(ValueError, match="symmetric"):
        ft.monitoring_errors(0.47, T4, N4, 1e-3,
                             index_cov=np.triu(np.ones((4, 4))))
    with pytest.raises(ValueError, match="semi-definite"):
        ft.monitoring_errors(0.47, T4, N4, 1e-3, index_cov=-np.eye(4))
    with pytest.raises(ValueError, match="index_errors"):
        ft.simulate_monitored_runs(0.47, T4, N4, 1e-3, 2,
                                   index_errors=np.zeros((3, 4)))
    with pytest.raises(ValueError, match="zero or negative"):
        ft.simulate_monitored_runs(0.47, T4, N4, 1e-3, 2,
                                   index_errors=[0.0, -2.0, 0.0, 0.0])


# ------------------ monitoring through an absorbing plate ------------------


def _inc_tmm_T(byrnes, lam, t, n, nsub, d_um):
    """T of the coated plate from `tmm.inc_tmm`, light from vacuum, s."""
    r = byrnes.inc_tmm("s", [1.0] + list(n) + [nsub, 1.0],
                       [np.inf] + list(np.asarray(t) * 1000)
                       + [d_um * 1000, np.inf],
                       ["i"] + ["c"] * len(t) + ["i", "i"], 0.0, lam * 1000)
    return r["T"]


def test_monitoring_through_an_absorbing_plate():
    """A monitor that measures through a 1 mm plate of index 1.52 + 1e-5 i
    (one pass keeps 77 % at 0.47 um): the cut-offs equal the
    transmittance of each partial stack from the independent `tmm`
    package; the slopes equal central differences of `stack_rt`; the
    simulated runs equal brentq on `stack_rt`; and the propagation
    matrix equals central differences of the simulated runs."""
    byrnes = pytest.importorskip("tmm")
    lam, ns, d = 0.47, 1.52 + 1e-5j, 1000.0
    plate = dict(n_sub=ns, n_exit=1.0, sub_thickness_um=d)
    p = ft.monitoring_plan(lam, T4, N4, sigma_signal=1e-3, **plate)
    worst = 0.0
    for j in range(4):
        ref = _inc_tmm_T(byrnes, lam, T4[j:], N4[j:], ns, d)
        worst = max(worst, abs(p.cutoff[j] - ref))
        h = 1e-7
        up = np.concatenate([[T4[j] + h], T4[j + 1:]])
        dn = np.concatenate([[T4[j] - h], T4[j + 1:]])
        fd = (ft.transmittance([lam], up, N4[j:], 1.0, **plate)[0]
              - ft.transmittance([lam], dn, N4[j:], 1.0, **plate)[0]) / (2 * h)
        assert p.slope[j] == pytest.approx(fd, rel=1e-7)   # measured 6.5e-9
    assert worst <= 1e-12                                  # measured 8e-16
    lossless = ft.monitoring_plan(lam, T4, N4, sigma_signal=1e-3,
                                  n_sub=1.52, n_exit=1.0)
    assert np.all(p.cutoff < lossless.cutoff)              # the plate absorbs

    e = np.random.default_rng(5).normal(0.0, 1e-3, (4, 4))
    got = ft.simulate_monitored_runs(lam, T4, N4, noise=e, **plate)
    for r in range(4):
        ref = _brentq_run(lam, T4, N4, e[r], ns, n_exit=1.0, d=d)
        assert np.max(np.abs(got[r] - ref)) <= 1e-12      # measured 1.4e-15
    me = ft.monitoring_errors(lam, T4, N4, 1e-3, **plate)
    h = 1e-6
    E = h * np.eye(4)
    rp = ft.simulate_monitored_runs(lam, T4, N4, noise=E, **plate)
    rm = ft.simulate_monitored_runs(lam, T4, N4, noise=-E, **plate)
    _close((rp - rm).T / (2 * h), me.propagation, rel=1e-6)  # measured 6.5e-9
    with pytest.raises(ValueError, match="needs n_exit"):
        ft.monitoring_plan(lam, T4, N4, n_sub=1.52, sub_thickness_um=d)
    with pytest.raises(ValueError, match="needs its thickness"):
        ft.monitoring_errors(lam, T4, N4, 1e-3, n_sub=ns, n_exit=1.0)


# ------------------ model= scoring, all designs at once ------------------


def _random_model(rng, N, L):
    C = int(rng.integers(1, 4))
    conds = [(float(rng.uniform(0, 1.2)), str(rng.choice(["s", "p", "u"])))
             for _ in range(C)]
    kind = int(rng.integers(0, 4))
    if kind == 0:
        m = ft.LinearMerit(rng.normal(size=(C, L)), 0.3,
                           str(rng.choice(["R", "T", "A"])))
    elif kind == 1:
        m = ft.TargetMerit(rng.uniform(0, 1, (C, L)),
                           str(rng.choice(["R", "T"])))
    elif kind == 2:
        stop = np.zeros(L, bool)
        stop[: max(1, L // 3)] = True
        m = ft.SpecMarginMerit(stop, ~stop, 0.2, 0.7, 50.0)
    else:
        def f(R, T, A):
            return (float(np.sum(T ** 2) - np.sum(R * A)), -A, 2 * T, -R)
        m = ft.FunctionMerit(f)
    kext = [None, rng.uniform(0, 0.05, L), rng.uniform(0, 0.05, (N, L))][
        int(rng.integers(0, 3))]
    if rng.random() < 0.4:
        return ft.OpticalModel(m, conds, kext=kext, n_exit=1.0,
                               sub_thickness_um=float(rng.uniform(200,
                                                                  2000)))
    return ft.OpticalModel(m, conds, kext=kext)


def test_batched_model_merits_equal_the_per_design_loop(monkeypatch):
    """`merits._model_merits_batch` against one `model_spectra` and one
    `model_merit_and_grad` call per design (the 0.12.0 loop), on 30
    random cases: 1 to 8 layers, 1 to 3 conditions (s, p, unpolarized,
    up to 69 degrees), every kind of merit, absorbing layers, absorbing
    plates with a back face, shared and per-layer dispersion, and small
    chunks (several per call)."""
    rng = np.random.default_rng(13)
    worst = 0.0
    for case in range(30):
        monkeypatch.setattr(merits, "_BATCH_COLUMNS",
                            53 if case % 3 == 0 else 2 ** 15)
        N = int(rng.integers(1, 9))
        L = int(rng.integers(3, 30))
        K = int(rng.integers(1, 25))
        lam = np.linspace(0.45, 0.8, L)
        S = ft.dispersion_shape(lam, 0.55)
        if case % 2:
            S = np.vstack([S] * N) * rng.uniform(0.97, 1.03, (N, 1))
        model = _random_model(rng, N, L)
        ns = 1.5 + 1e-5j if model.n_exit is not None else \
            [1.46, ft.SIO2_MALITSON1965.n(lam)][case % 2]
        tk = rng.uniform(0.02, 0.2, (K, N))
        nk = rng.uniform(1.4, 2.4, (K, N))
        J, gt, gn = merits._model_merits_batch(model, lam, tk, nk, S, 1.0,
                                               ns, grads=True)
        ref = [model_merit_and_grad(model, lam, tk[k], nk[k], S, 1.0, ns)
               for k in range(K)]
        worst = max(worst, _close(J, [r[0] for r in ref]),
                    _close(gt, [r[1] for r in ref]),
                    _close(gn, [r[2] for r in ref]))
        J0 = merits._model_merits_batch(model, lam, tk, nk, S, 1.0, ns)
        Jl = [model.merit(*model_spectra(model, lam, tk[k], nk[k], S, 1.0,
                                         ns))[0] for k in range(K)]
        worst = max(worst, _close(J0, Jl))
    assert worst <= 1e-12        # measured 0 with gradients, 6e-15 without


def _draws_loop(lam, t, n0, S, model, twin, z, box, n_sub):
    """The per-draw loop of 0.12.0 (fabtwin.robust._draws, model=)."""
    N = t.size
    x = twin.transform(z)
    t_lo, t_hi, n_lo, n_hi = robust._fab_clip_bounds(box)
    J = np.empty(len(x))
    gT = np.empty((len(x), N))
    gN = np.empty((len(x), N))
    for k in range(len(x)):
        tk_raw = t * (1.0 + x[k, :N])
        nk_raw = n0 + x[k, N:]
        Jk, gdk, gnk = model_merit_and_grad(
            model, lam, np.clip(tk_raw, t_lo, t_hi),
            np.clip(nk_raw, n_lo, n_hi), S, n_sub=n_sub)
        J[k] = Jk
        gT[k] = gdk * (1.0 + x[k, :N]) * ((tk_raw > t_lo) & (tk_raw < t_hi))
        gN[k] = gnk * ((nk_raw > n_lo) & (nk_raw < n_hi))
    return J, gT, gN


def test_model_scoring_and_robust_design_equal_the_0_12_loops():
    """`evaluate_under_process`, `induced_merits`, `random_search`, the
    probes of `inverse_design` and the draws of `cvar_objective_and_grad`
    / `ru_objective_and_grad` with model=, against the 0.12.0 per-design
    loops (1e-12 of the largest value; measured at most 5e-16)."""
    lam = np.linspace(0.45, 0.65, 31)
    S = ft.dispersion_shape(lam, 0.55)
    w, c = ft.notch_weights(lam, 0.532, 0.012, 0.02)
    model = ft.OpticalModel(ft.LinearMerit(np.tile(w / 2, (2, 1)), c, "T"),
                            ((0.0, "u"), (0.35, "p")), kext=0.01 * S)
    box = ft.DesignBox(6, 0.03, 0.15, 1.7, 2.3)
    rng = np.random.default_rng(6)
    t, n = (a[0] for a in box.sample(rng, 1))
    x = rng.normal(0.0, 0.02, (60, 12))
    tw = ft.GaussianTwin(x)

    def loop(tt, nn):
        return np.array([model.merit(*model_spectra(model, lam, tt[k], nn[k],
                                                    S, 1.0, 1.46))[0]
                         for k in range(len(tt))])

    def sampler(a, b, K):
        return tw.sample(np.random.default_rng(8), a, b, K)

    out = ft.evaluate_under_process(sampler, t, n, lam, S, None, None, 200,
                                    n_sub=1.46, alpha=0.1, model=model)
    _close(out["samples"], loop(*sampler(t, n, 200)))
    tt, nn = ft.apply_errors(t, n, x)
    _close(ft.induced_merits(x, t, n, lam, S, None, None, n_sub=1.46,
                             model=model), loop(tt, nn))
    _, _, Jb, hist = ft.random_search(lam, S, None, None, box, 80,
                                      n_sub=1.46, seed=4, model=model)
    _close(hist, np.maximum.accumulate(loop(*box.sample(
        np.random.default_rng(4), 80))))
    _, _, _, traj = ft.inverse_design(lam, S, None, None, box, n_probe=50,
                                      n_seed=1, n_iter=2, n_sub=1.46,
                                      seed=2, model=model)
    _close(traj[:50], np.maximum.accumulate(loop(*box.sample(
        np.random.default_rng(2), 50))))

    z = rng.normal(size=(40, 12))
    J, gT, gN = _draws_loop(lam, t, n, S, model, tw, z, box, 1.46)
    v, gt, gn, Jc = ft.cvar_objective_and_grad(lam, t, n, S, None, None, tw,
                                               z, 0.1, box, n_sub=1.46,
                                               model=model)
    _close(Jc, J)
    wgt = np.zeros(40)
    wgt[np.argsort(J)[:4]] = 0.25
    _close(gt, wgt @ gT)
    _close(gn, wgt @ gN)
    tau = float(np.quantile(J, 0.2))
    _, gt, gn, _, _ = ft.ru_objective_and_grad(lam, t, n, tau, S, None,
                                               None, tw, z, 0.2, box,
                                               n_sub=1.46, model=model)
    below = (J < tau) / (0.2 * 40)
    _close(gt, below @ gT)
    _close(gn, below @ gN)


# ------------------------------ fixes ------------------------------


def test_negative_thicknesses_and_wavelengths_are_refused():
    """0.12.0 returned T = 1.208 (more light out than in) for an
    absorbing layer of index 2.0 + 0.3i given a thickness of -0.05 um on
    1.5 at 0.5 um, and the same for a wavelength of -0.5 um; the closed
    form check: the same layer at +0.05 um absorbs (T < 1)."""
    lam = np.array([0.5])
    _, T = ft.stack_rt(lam, [0.05], [2.0 + 0.3j], 1.0, 1.5)
    assert T[0] < 1.0
    for bad_t, bad_lam in (([-0.05], lam), ([0.05], -lam),
                           ([np.nan], lam), ([0.05], [0.0]),
                           ([0.05], [np.inf])):
        with pytest.raises(ValueError, match="thickness|wavelength"):
            ft.stack_rt(bad_lam, bad_t, [2.0 + 0.3j], 1.0, 1.5)
        with pytest.raises(ValueError, match="thickness|wavelength"):
            ft.stack_rta_and_grads(np.asarray(bad_lam, float),
                                   np.asarray(bad_t, float),
                                   [2.0 + 0.3j], 1.0, 1.5)
        with pytest.raises(ValueError, match="thickness|wavelength"):
            ft.transmittance_and_grads(bad_lam, bad_t, [2.0], np.ones(1),
                                       n_sub=1.5)
    with pytest.raises(ValueError, match="thickness"):
        ft.transmittance(lam, [0.05, -0.01], [2.0, 1.5], 1.0, 1.5,
                         n_exit=1.0)
    # a layer of zero thickness is allowed and changes nothing
    R0, T0 = ft.stack_rt(lam, [0.0], [2.0 + 0.3j], 1.0, 1.5)
    Rb, Tb = ft.stack_rt(lam, [], np.zeros((0, 1)), 1.0, 1.5)
    assert abs(R0[0] - Rb[0]) < 1e-15 and abs(T0[0] - Tb[0]) < 1e-15


def test_spectrum_fits_keep_thicknesses_positive():
    """With a search box wider than the thicknesses (max_error=1.0, box
    +/- 2), 0.12.0 let the fit pass through zero thickness. For this
    noise-free spectrum it returned (0.289, 0.201, -1.097), a last layer
    of -0.006 um, with seed 1; for a second case it refused a unique
    answer as non-unique, and `errors_from_spectra` returned -1.013.
    Now the thicknesses stay positive and the true errors come back."""
    lam = np.linspace(0.4, 0.8, 60)
    S = np.ones(60)
    t0 = np.array([0.124, 0.135, 0.064])
    n0 = np.array([1.46, 2.1, 1.46])
    for xt, seed in (([0.289, 0.206, -0.902], 1),
                     ([-0.276, 0.139, -0.987], 1)):
        xt = np.array(xt)
        T, _, _ = ft.transmittance_and_grads(lam, t0 * (1 + xt), n0, S,
                                             n_sub=1.5)
        r = ft.errors_from_spectrum(lam, T, t0, n0, S, n_sub=1.5,
                                    max_error=1.0, seed=seed)
        assert np.all(r.t_um > 0)
        assert np.max(np.abs(r.dt_over_t - xt)) <= 1e-6   # measured 8e-13
        j = ft.errors_from_spectra(lam, [ft.Measurement(0.0, "s", "T", T)],
                                   t0, n0, S, n_sub=1.5, max_error=1.0,
                                   seed=0)
        assert np.max(np.abs(j.dt_over_t - xt)) <= 1e-6   # measured 1.4e-11
    # the usual box (max_error=0.1) is not affected: same numbers as the
    # bounds +/- 0.2 written out
    xt = np.array([0.03, -0.02, 0.05])
    T, _, _ = ft.transmittance_and_grads(lam, t0 * (1 + xt), n0, S,
                                         n_sub=1.5)
    r = ft.errors_from_spectrum(lam, T, t0, n0, S, n_sub=1.5)
    assert np.max(np.abs(r.dt_over_t - xt)) <= 1e-8    # measured 5e-14


def test_robust_objectives_refuse_alpha_outside_0_1_and_empty_batches():
    """With alpha = 1.5, 0.12.0 returned the mean of all draws as the
    value but 2/3 of its derivative as the gradient; with alpha <= 0 the
    single worst draw; with K = 0, `robustify` returned the start design
    unchanged after NaN objectives. At alpha = 1 the gradient is the
    derivative of the value (central finite difference)."""
    lam = np.linspace(0.45, 0.65, 21)
    S = np.ones(21)
    w, c = ft.notch_weights(lam, 0.55, 0.02, 0.02)
    box = ft.DesignBox(4, 0.03, 0.15, 1.5, 2.3)
    t = np.array([0.08, 0.10, 0.07, 0.12])
    n = np.array([2.1, 1.5, 2.1, 1.5])
    tw = ft.GaussianTwin(np.random.default_rng(0).normal(0, 0.02, (30, 8)))
    z = np.random.default_rng(1).normal(size=(20, 8))

    def val(tt):
        return ft.cvar_objective_and_grad(lam, tt, n, S, w, c, tw, z, 1.0,
                                          box, n_sub=1.46)[0]
    v, gt, _, J = ft.cvar_objective_and_grad(lam, t, n, S, w, c, tw, z, 1.0,
                                             box, n_sub=1.46)
    assert v == pytest.approx(J.mean(), rel=1e-15)
    h = 1e-6
    e = np.array([h, 0, 0, 0])
    assert gt[0] == pytest.approx((val(t + e) - val(t - e)) / (2 * h),
                                  rel=1e-6)          # measured 8.5e-9
    for a in (1.5, 0.0, -0.1, np.nan):
        with pytest.raises(ValueError, match="alpha"):
            ft.cvar_objective_and_grad(lam, t, n, S, w, c, tw, z, a, box)
        with pytest.raises(ValueError, match="alpha"):
            ft.robustify(lam, t, n, S, w, c, tw, box, alpha=a, steps=1)
    with pytest.raises(ValueError, match="alpha"):
        ft.ru_objective_and_grad(lam, t, n, 0.5, S, w, c, tw, z, 1.5, box)
    with pytest.raises(ValueError, match="K must be"):
        ft.robustify(lam, t, n, S, w, c, tw, box, K=0, steps=1)
    with pytest.raises(ValueError, match="draws"):
        ft.cvar_objective_and_grad(lam, t, n, S, w, c, tw, z[:0], 0.1, box)
    # mean-variance does not use alpha
    ft.cvar_objective_and_grad(lam, t, n, S, w, c, tw, z, 7.0, box,
                               mean_variance=True)


def test_conditioning_on_components_of_zero_variance():
    """The twin of a monitored run has index errors of zero variance.
    0.12.0 raised LinAlgError when it was conditioned on them (and so
    `reoptimize_remaining` could not use it). Now the result equals the
    twin conditioned on the thickness components alone (those index
    errors carry no information), with the index components fixed at
    the measured values."""
    me = ft.monitoring_errors(0.47, T4, N4, 1e-3, n_sub=1.52)
    tw = me.twin()
    obs = [2, 3, 6, 7]
    val = [0.01, -0.005, 0.003, 0.0]
    c = tw.conditional(obs, val)
    ref = tw.conditional([2, 3], [0.01, -0.005])
    _close(c.mu[:2], ref.mu[:2], rel=1e-12)          # measured 0
    _close(c.cov[:2, :2], ref.cov[:2, :2], rel=1e-12)  # measured 3e-15
    assert np.array_equal(c.mu[obs], val)
    assert np.all(c.cov[obs] == 0.0)
    # by hand: mu_u + C_uo C_oo^-1 (x_o - mu_o) on the thickness block
    C = tw.cov[:4, :4]
    k = np.linalg.solve(C[2:, 2:], C[2:, :2]).T
    _close(c.mu[:2], tw.mu[:2] + k @ (np.array(val[:2]) - tw.mu[2:4]),
           rel=1e-12)                                # measured 0
    # and the correction of a run can now use the monitored machine
    lam = np.linspace(0.45, 0.65, 21)
    S = np.ones(21)
    w, cc = ft.notch_weights(lam, 0.55, 0.02, 0.02)
    box = ft.DesignBox(4, 0.03, 0.15, 1.4, 2.2)
    out = ft.reoptimize_remaining(lam, T4, N4, S, w, cc, 2, T4[2:] * 1.01,
                                  N4[2:], box, twin=tw, K=16, steps=5,
                                  n_iter=5, n_sub=1.52)
    assert np.array_equal(out["t"][2:], T4[2:] * 1.01)
    assert np.all(np.isfinite(out["t"]))


def test_reference_process_refuses_a_rate_bias_of_minus_one():
    """beta_t = -1 deposits nothing and below -1 negative thicknesses;
    0.12.0 returned them (zero thicknesses from `deterministic_map`)."""
    import dataclasses
    for b in (-1.0, -1.5):
        with pytest.raises(ValueError, match="beta_t"):
            dataclasses.replace(ft.PAPER_PROCESS, beta_t=b)
    ok = dataclasses.replace(ft.PAPER_PROCESS, beta_t=-0.5)
    t, _ = ok.deterministic_map([0.1, 0.2], [2.0, 1.5])
    _close(t, [0.05, 0.1], rel=1e-15)  # measured 0

