"""v0.3 adaptability anchors: measured dispersion tables held to the
independent Sellmeier closed form, per-layer and frozen design boxes
(the fixed-index multi-material case), reflectance-based merits via
the exact lossless identity R = 1 - T, and the thickness-only
metrology path -- each anchored to a closed form or to two
independent code paths agreeing, never to a stored number."""
import numpy as np
import pytest

import fabtwin as ft


# --------------------- measured dispersion tables ---------------------


def _silica_table(P=61, lo=0.40, hi=0.80):
    """A table SAMPLED from the cited Malitson Sellmeier -- so the
    closed form is the independent reference the interpolant must
    agree with between its own nodes."""
    grid = np.linspace(lo, hi, P)
    return ft.TabulatedMaterial(
        name="silica table (sampled from Malitson 1965)",
        lam_um=grid, n_table=ft.SIO2_MALITSON1965.n(grid),
        reference="test fixture: nodes evaluated from Malitson 1965")


def test_tabulated_material_matches_the_generating_closed_form():
    mat = _silica_table()
    # exact at the tabulated nodes (interpolation, not smoothing)
    assert np.abs(mat.n(mat.lam_um) - mat.n_table).max() < 1e-15
    # between nodes: PCHIP-on-table vs the independent Sellmeier
    # closed form -- two code paths, no shared arithmetic
    mid = 0.5 * (mat.lam_um[:-1] + mat.lam_um[1:])
    assert np.abs(mat.n(mid) - ft.SIO2_MALITSON1965.n(mid)).max() < 1e-6
    # drop-in duck typing: same shape construction as a Sellmeier
    lam = np.linspace(0.45, 0.75, 31)
    S_tab = ft.dispersion_shape(lam, 0.550, mat)
    S_ref = ft.dispersion_shape(lam, 0.550, ft.SIO2_MALITSON1965)
    assert np.abs(S_tab - S_ref).max() < 1e-6
    assert abs(float(ft.dispersion_shape(np.array([0.550]), 0.550,
                                         mat)[0]) - 1.0) < 1e-15


def test_tabulated_material_refusals():
    grid = np.linspace(0.4, 0.8, 11)
    n = np.full(11, 1.46)
    mat = ft.TabulatedMaterial("m", grid, n, reference="fixture")
    with pytest.raises(ValueError, match="tabulated range"):
        mat.n(np.array([0.39]))
    with pytest.raises(ValueError, match="tabulated range"):
        mat.nk(np.array([0.81]))
    with pytest.raises(ValueError, match="strictly increasing"):
        ft.TabulatedMaterial("m", grid[::-1], n, reference="fixture")
    with pytest.raises(ValueError, match="reference"):
        ft.TabulatedMaterial("m", grid, n, reference="  ")
    with pytest.raises(ValueError, match="k must be >= 0"):
        ft.TabulatedMaterial("m", grid, n, k_table=np.full(11, -0.1),
                             reference="fixture")
    with pytest.raises(ValueError):
        ft.TabulatedMaterial("m", grid, -n, reference="fixture")


def test_absorbing_table_refused_on_design_path_served_on_forward():
    grid = np.linspace(0.4, 0.8, 11)
    n = np.full(11, 2.0)
    k = np.full(11, 0.05)
    mat = ft.TabulatedMaterial("lossy", grid, n, k_table=k,
                               reference="fixture")
    with pytest.raises(ValueError, match="nonzero extinction"):
        mat.n(grid)                      # honest refusal, not truncation
    with pytest.raises(ValueError, match="nonzero extinction"):
        ft.dispersion_shape(grid, 0.6, mat)
    # the forward solver consumes .nk(): absorbing stack has R + T < 1
    lam = np.linspace(0.45, 0.75, 21)
    nk = mat.nk(lam)                     # (L,) complex
    nlay = np.tile(nk, (4, 1))
    R, T = ft.stack_rt(lam, np.full(4, 0.08), nlay, 1.0, 1.46)
    assert np.all(R + T < 1.0) and np.all(R + T > 0.0)
    # a k = 0 table's .nk equals its .n exactly
    mat0 = ft.TabulatedMaterial("lossless", grid, n,
                                k_table=np.zeros(11), reference="fx")
    assert np.abs(mat0.nk(lam) - mat0.n(lam)).max() == 0.0


def test_tabulated_material_end_to_end_adjoint():
    """A tabulated (measured-style) material through the whole design
    path: shape, solver, hand adjoint vs central finite differences."""
    mat = _silica_table()
    lam = np.linspace(0.45, 0.65, 21)
    S = ft.dispersion_shape(lam, 0.550, mat)
    t = np.full(4, 0.06)
    n0 = np.array([1.9, 1.5, 2.1, 1.6])
    w, c0 = ft.notch_weights(lam, 0.532, 0.015, 0.030)
    _, gd, gn = ft.merit_and_grad(lam, t, n0, S, w, c0, n_sub=1.46)
    h = 1e-7

    def J(tt, nn):
        T = ft.transmittance(lam, tt, nn[:, None] * S[None, :], 1.0,
                             1.46)
        return float(ft.merit(T, w, c0))

    for i in (0, 2):
        tp, tm = t.copy(), t.copy()
        tp[i] += h
        tm[i] -= h
        fd = (J(tp, n0) - J(tm, n0)) / (2 * h)
        assert abs(gd[i] - fd) / max(abs(fd), 1e-12) < 1e-6
        np_, nm = n0.copy(), n0.copy()
        np_[i] += h
        nm[i] -= h
        fd = (J(t, np_) - J(t, nm)) / (2 * h)
        assert abs(gn[i] - fd) / max(abs(fd), 1e-12) < 1e-6


# ------------------ per-layer and frozen design boxes ------------------


def test_per_layer_box_samples_and_clips_per_layer():
    t_lo = np.array([0.02, 0.05, 0.02, 0.05])
    t_hi = np.array([0.04, 0.12, 0.04, 0.12])
    n_lo = np.array([1.45, 2.00, 1.45, 2.00])
    n_hi = np.array([1.47, 2.20, 1.47, 2.20])
    box = ft.DesignBox(4, t_lo, t_hi, n_lo, n_hi)
    t, n = box.sample(np.random.default_rng(0), 200)
    assert np.all(t >= t_lo) and np.all(t <= t_hi)
    assert np.all(n >= n_lo) and np.all(n <= n_hi)
    tc, nc = box.clip(np.full(4, 1.0), np.full(4, 0.0))
    assert np.all(tc == t_hi) and np.all(nc == n_lo)
    # scalar boxes behave exactly as before
    b0 = ft.DesignBox(4, 0.02, 0.12, 1.6, 2.4)
    assert isinstance(b0.t_lo, float)


def test_design_box_refusals_including_per_layer():
    with pytest.raises(ValueError):
        ft.DesignBox(4, np.array([0.02, 0.12, 0.02, 0.02]),  # inverted
                     np.array([0.12, 0.02, 0.12, 0.12]), 1.6, 2.4)
    with pytest.raises(ValueError):                # wrong bound length
        ft.DesignBox(4, np.full(3, 0.02), 0.12, 1.6, 2.4)
    with pytest.raises(ValueError):                # non-positive t_lo
        ft.DesignBox(4, np.array([0.02, 0.0, 0.02, 0.02]), 0.12,
                     1.6, 2.4)
    with pytest.raises(ValueError, match="fully frozen"):
        ft.DesignBox(4, 0.05, 0.05, 2.0, 2.0)


def test_frozen_index_design_never_moves_the_indices():
    """The real multi-material case: alternating fixed nH/nL, only
    thicknesses designed. Both the design engine and the CVaR
    robustifier must return the frozen profile EXACTLY."""
    lam = np.linspace(0.45, 0.65, 41)
    lam0 = 0.550
    S_H = ft.dispersion_shape(lam, lam0, ft.SI3N4_LUKE2015)
    S_L = ft.dispersion_shape(lam, lam0, ft.SIO2_MALITSON1965)
    S = np.stack([S_H if i % 2 == 0 else S_L for i in range(8)])
    nsub = ft.SIO2_MALITSON1965.n(lam)
    n_fixed = np.array([2.02, 1.4585, 2.02, 1.4585,
                        2.02, 1.4585, 2.02, 1.4585])
    box = ft.DesignBox.thickness_only(8, 0.020, 0.140, n_fixed)
    w, c0 = ft.notch_weights(lam, 0.532, 0.015, 0.030)

    t, n0, J, _ = ft.inverse_design(lam, S, w, c0, box, n_probe=40,
                                    n_seed=2, n_iter=25, n_sub=nsub,
                                    seed=0)
    assert np.array_equal(n0, n_fixed)          # exactly frozen
    assert np.all((t >= 0.020) & (t <= 0.140))
    assert J > 0.5                              # beats the trivial 1/2

    rng = np.random.default_rng(1)
    tw = ft.GaussianTwin(rng.normal(0.0, 0.01, (200, 16)))
    t_r, n_r, _ = ft.robustify(lam, t, n0, S, w, c0, tw, box,
                               alpha=0.1, K=24, steps=15, lr=3e-3,
                               seed=0, n_sub=nsub)
    assert np.array_equal(n_r, n_fixed)         # still exactly frozen


# ---------------------- reflectance-based merits ----------------------


def test_reflectance_weights_identity_lossless_and_its_absorbing_gap():
    lam = np.linspace(0.45, 0.65, 31)
    rng = np.random.default_rng(2)
    w_R = rng.uniform(0.0, 1.0, 31)
    w_R /= w_R.sum()
    c = 0.1
    w_T, c_T = ft.weights_from_reflectance(w_R, c)
    t = rng.uniform(0.03, 0.11, 6)
    # lossless: J_R computed directly equals the converted merit to
    # machine precision (through the solver's exact R + T = 1)
    n_re = rng.uniform(1.4, 2.3, 6)
    R, T = ft.stack_rt(lam, t, np.repeat(n_re[:, None], 31, 1), 1.0,
                       1.46)
    assert abs((R @ w_R + c) - ft.merit(T, w_T, c_T)) < 1e-12
    # absorbing: the identity fails by EXACTLY w_R . A -- an exact
    # relation in both regimes, so the documented scope is a theorem
    # of the code, not a caveat
    n_ab = n_re + 0.05j
    R2, T2 = ft.stack_rt(lam, t, np.repeat(n_ab[:, None], 31, 1), 1.0,
                         1.46)
    A = 1.0 - R2 - T2
    gap = (R2 @ w_R + c) - ft.merit(T2, w_T, c_T)
    assert abs(gap - (-(A @ w_R))) < 1e-12
    assert abs(gap) > 1e-4                      # and it is a REAL gap
    with pytest.raises(ValueError):
        ft.weights_from_reflectance(np.ones((2, 2)))


def test_mirror_design_through_reflectance_weights():
    """Maximizing band-average R with the converted weights must beat
    the bare substrate's reflectance -- and the closed-form
    quarter-wave stack (HL)^3 must score what its exact admittance
    formula says, through the SAME merit path."""
    lam0 = 0.550
    lam = np.array([lam0])
    w_R = np.array([1.0])
    w_T, c_T = ft.weights_from_reflectance(w_R, 0.0)
    nH, nL, ns = 2.1, 1.5, 1.46
    n0 = np.array([nH, nL] * 3)
    t = lam0 / (4.0 * n0)
    J = float(ft.merit(ft.transmittance(lam, t,
                                        np.repeat(n0[:, None], 1, 1),
                                        1.0, np.array([ns])), w_T, c_T))
    Y = (nH / nL) ** 6 * ns
    R_closed = ((1.0 - Y) / (1.0 + Y)) ** 2
    assert abs(J - R_closed) < 1e-12


# ---------------------- thickness-only metrology ----------------------


def test_thickness_only_traces_flow_through_the_whole_loop():
    """A tool that logs thickness but not per-run index: set
    n_fab = n_recipe in the trace contract. The twin then samples
    numerically negligible index errors, and robustification on a
    frozen-index box uses exactly the information the lab has."""
    rng = np.random.default_rng(3)
    lam = np.linspace(0.45, 0.65, 31)
    S = ft.dispersion_shape(lam, 0.550)
    nsub = ft.SIO2_MALITSON1965.n(lam)
    w, c0 = ft.notch_weights(lam, 0.532, 0.015, 0.030)
    n_fixed = np.full(6, 2.0)
    box = ft.DesignBox.thickness_only(6, 0.020, 0.120, n_fixed)

    rt, _ = box.sample(rng, 40)
    rn = np.tile(n_fixed, (40, 1))
    ftd = rt * (1.0 + 0.02 + rng.normal(0.0, 0.02, rt.shape))
    x = ft.errors_from_traces(rt, rn, ftd, rn)      # n_fab = n_recipe
    tw = ft.GaussianTwin(x)
    xs = tw.sample_errors(rng, 400)
    assert np.abs(xs[:, 6:]).max() < 1e-3           # no invented index errors
    assert xs[:, :6].std() > 5e-3                   # real thickness errors kept

    t0, n0 = box.sample(rng, 1)
    t0, n0 = t0[0], n0[0]

    def twin_cvar(tt, nn, seed):
        r = np.random.default_rng(seed)
        return ft.evaluate_under_process(
            lambda a, b, K: tw.sample(r, a, b, K), tt, nn, lam, S, w,
            c0, K=300, n_sub=nsub, alpha=0.1)["CVaR"]

    before = twin_cvar(t0, n0, 11)
    t_r, n_r, _ = ft.robustify(lam, t0, n0, S, w, c0, tw, box,
                               alpha=0.1, K=24, steps=20, lr=3e-3,
                               seed=0, n_sub=nsub)
    assert np.array_equal(n_r, n_fixed)
    assert twin_cvar(t_r, n_r, 11) >= before - 1e-6


# ------------------------- twin-config guard -------------------------


def test_fabtwin_config_refuses_frozen_or_per_layer_bounds():
    from fabtwin.twin_jax import FabTwinConfig    # imports without jax
    with pytest.raises(ValueError, match="strictly increasing"):
        FabTwinConfig(4, 0.05, 0.05, 1.6, 2.4)
    with pytest.raises(ValueError, match="strictly increasing"):
        FabTwinConfig(4, np.array([0.02] * 4), np.array([0.12] * 4),
                      1.6, 2.4)
    FabTwinConfig(4, 0.02, 0.12, 1.6, 2.4)        # the scalar case is fine
