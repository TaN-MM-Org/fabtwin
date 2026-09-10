"""Real-data pathway anchors: the trace-file contract round-trips
exactly and refuses malformed input; validation separates structural
refusals from plausibility flags; the fidelity metrics satisfy their
exact identities (zero against self, W1 equal to SciPy's independent
implementation, risk-module tail conventions); and the complete
bring-your-own-data pipeline -- CSV in, validate, split, fit, score
against the held-out half, robustify -- runs end to end with the
reference process standing in for a lab's tool."""
import dataclasses

import numpy as np
import pytest
from scipy.stats import wasserstein_distance

import fabtwin as ft

RNG = np.random.default_rng(0)
T_REC = np.linspace(0.030, 0.100, 8)
N_REC = np.linspace(1.70, 2.30, 8)


def _traces(n_recipes=30, runs=2, seed=1):
    rng = np.random.default_rng(seed)
    box = ft.DesignBox(8, 0.020, 0.120, 1.6, 2.4)
    rec = box.sample(rng, n_recipes)
    return ft.PAPER_PROCESS.trace_dataset(*rec, runs, rng)


# ----------------------------- data -------------------------------


def test_csv_round_trip_is_exact(tmp_path):
    rt, rn, ftd, fnd = _traces(5, 2)
    path = tmp_path / "traces.csv"
    ft.save_traces_csv(path, rt, rn, ftd, fnd)
    rt2, rn2, ft2, fn2 = ft.load_traces_csv(path)
    assert np.array_equal(rt, rt2) and np.array_equal(rn, rn2)
    assert np.array_equal(ftd, ft2) and np.array_equal(fnd, fn2)


def test_loader_refuses_malformed_files(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("wrong,header\n1,2\n")
    with pytest.raises(ValueError):
        ft.load_traces_csv(p)
    p.write_text("run,layer,t_recipe_um,n_recipe,t_fab_um,n_fab\n")
    with pytest.raises(ValueError):
        ft.load_traces_csv(p)                     # no data rows
    p.write_text("run,layer,t_recipe_um,n_recipe,t_fab_um,n_fab\n"
                 "1,1,0.05,2.0,0.051,2.01\n"
                 "1,3,0.05,2.0,0.051,2.01\n")     # gap in layers
    with pytest.raises(ValueError):
        ft.load_traces_csv(p)
    p.write_text("run,layer,t_recipe_um,n_recipe,t_fab_um,n_fab\n"
                 "1,1,0.05,2.0,0.051,2.01\n"
                 "1,1,0.05,2.0,0.051,2.01\n")     # duplicate layer
    with pytest.raises(ValueError):
        ft.load_traces_csv(p)
    p.write_text("run,layer,t_recipe_um,n_recipe,t_fab_um,n_fab\n"
                 "1,1,0.05,2.0,0.051,2.01\n"
                 "2,1,0.05,2.0,0.051,2.01\n"
                 "2,2,0.05,2.0,0.051,2.01\n")     # ragged runs
    with pytest.raises(ValueError):
        ft.load_traces_csv(p)


def test_validation_raises_on_structure_and_flags_plausibility():
    rt, rn, ftd, fnd = _traces(4, 1)
    rep = ft.validate_traces(rt, rn, ftd, fnd)
    assert rep["n_runs"] == 4 and rep["n_layers"] == 8
    assert rep["flags"] == []                    # nominal process data
    bad = ftd.copy()
    bad[1, 2] = np.nan
    with pytest.raises(ValueError):
        ft.validate_traces(rt, rn, bad, fnd)
    bad = ftd.copy()
    bad[0, 0] = -0.01
    with pytest.raises(ValueError):
        ft.validate_traces(rt, rn, bad, fnd)
    with pytest.raises(ValueError):
        ft.validate_traces(rt[:2], rn, ftd, fnd)
    # a unit mix-up (nm written where um belongs) is FLAGGED, not
    # raised and not dropped
    mixed = ftd.copy()
    mixed[2, 5] = ftd[2, 5] * 1000.0
    rep = ft.validate_traces(rt, rn, mixed, fnd)
    assert (2, 5, "rel_t", pytest.approx(mixed[2, 5] / rt[2, 5] - 1.0)) \
        in [tuple(f) for f in rep["flags"]]


# --------------------------- fidelity ------------------------------


def test_fidelity_distances_vanish_against_self():
    x = RNG.normal(0.0, 0.01, (150, 16))
    me = ft.moment_errors(x, x)
    assert me["d_mean"] == 0.0 and me["d_cov"] == 0.0 \
        and me["d_corr"] == 0.0
    j = RNG.normal(0.8, 0.05, 200)
    dd = ft.distribution_distances(j, j, 0.05)
    assert dd["d_mean"] == dd["d_P"] == dd["d_CVaR"] == dd["W1"] == 0.0


def test_w1_equals_scipy_and_tails_use_risk_conventions():
    a = RNG.normal(0.80, 0.05, 300)
    b = RNG.normal(0.75, 0.07, 250)
    dd = ft.distribution_distances(a, b, 0.1)
    assert dd["W1"] == wasserstein_distance(a, b)
    assert dd["d_CVaR"] == abs(ft.cvar(a, 0.1) - ft.cvar(b, 0.1))
    assert dd["d_P"] == abs(np.percentile(a, 10) - np.percentile(b, 10))


def test_mean_shifted_twin_scores_strictly_worse():
    base = RNG.normal(0.0, 0.01, (400, 16))
    fit = RNG.normal(0.0, 0.01, (400, 16))
    biased = fit + 0.05
    assert ft.moment_errors(biased, base)["d_mean"] > \
        ft.moment_errors(fit, base)["d_mean"]


def test_induced_merits_match_direct_solver_evaluation():
    lam = np.linspace(0.45, 0.65, 41)
    S = ft.dispersion_shape(lam, 0.550)
    nsub = ft.SIO2_MALITSON1965.n(lam)
    w, c0 = ft.notch_weights(lam, 0.532, 0.015, 0.030)
    x = RNG.normal(0.0, 0.01, (5, 16))
    J = ft.induced_merits(x, T_REC, N_REC, lam, S, w, c0, n_sub=nsub)
    tt, nn = ft.apply_errors(T_REC, N_REC, x)
    for k in range(5):
        T = ft.transmittance(lam, tt[k], nn[k][:, None] * S[None, :],
                             1.0, nsub)
        assert J[k] == float(ft.merit(T, w, c0))


# ------------------- the bring-your-own-data loop -------------------


def test_full_pipeline_from_csv_to_robust_design(tmp_path):
    """A lab's workflow end to end, with the reference process
    standing in for the tool: traces arrive as a CSV, are validated,
    split into fit/held-out halves, a twin is fitted and SCORED
    against the held-out half through the solver, and the design is
    robustified under it. Assertions are identities and orderings,
    not invented numbers."""
    lam = np.linspace(0.45, 0.65, 41)
    S = ft.dispersion_shape(lam, 0.550)
    nsub = ft.SIO2_MALITSON1965.n(lam)
    w, c0 = ft.notch_weights(lam, 0.532, 0.015, 0.030)
    box = ft.DesignBox(8, 0.020, 0.120, 1.6, 2.4)

    rt, rn, ftd, fnd = _traces(40, 2, seed=5)
    path = tmp_path / "fab_traces.csv"
    ft.save_traces_csv(path, rt, rn, ftd, fnd)
    rt, rn, ftd, fnd = ft.load_traces_csv(path)
    assert ft.validate_traces(rt, rn, ftd, fnd)["n_runs"] == 80

    x = ft.errors_from_traces(rt, rn, ftd, fnd)
    x_fit, x_hold = x[::2], x[1::2]
    twin = ft.GaussianTwin(x_fit)

    designs = [(rt[0], rn[0]), (rt[2], rn[2])]
    rep = ft.twin_fidelity_report(
        twin.sample_errors(np.random.default_rng(1), 400), x_hold,
        designs, lam, S, w, c0, n_sub=nsub, alpha=0.1)
    assert np.isfinite(rep["induced"]["W1"])
    assert len(rep["per_design"]) == 2
    # a twin fitted to the data beats a deliberately broken twin on
    # the held-out moments
    broken = ft.GaussianTwin(x_fit + 0.05)
    rep_b = ft.twin_fidelity_report(
        broken.sample_errors(np.random.default_rng(1), 400), x_hold,
        designs, lam, S, w, c0, n_sub=nsub, alpha=0.1)
    assert rep_b["moments"]["d_mean"] > rep["moments"]["d_mean"]
    assert rep_b["induced"]["W1"] > rep["induced"]["W1"]

    t0, n0 = rt[0], rn[0]
    t_r, n_r, _ = ft.robustify(lam, t0, n0, S, w, c0, twin, box,
                               alpha=0.1, K=24, steps=15, lr=3e-3,
                               seed=0, n_sub=nsub)
    assert np.all((t_r >= box.t_lo) & (t_r <= box.t_hi))
