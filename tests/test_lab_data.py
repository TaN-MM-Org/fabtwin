"""0.9.0: two bug fixes, input checks, the spectrophotometer reader
`load_spectrum_csv` and the measurement planner `recovery_precision`.

Independent references used here: exact rational arithmetic
(`fractions.Fraction`) for the tail counts and conformal ranks; the
mean of the smallest values computed directly; central finite
differences of the forward solver `fabtwin.tmm` (not the adjoint) for
the planner's Jacobian; the scatter of actual nonlinear fits to
simulated noisy spectra; and files written by the test itself with
known unit conversions."""
import re
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import fabtwin as ft
from fabtwin.risk import _ceil_count

ROOT = Path(__file__).resolve().parents[1]


def _exact_ceil(fr):
    return -(-fr.numerator // fr.denominator)


# ------------------- bug 1: tail counts and ranks -------------------


def test_tail_count_matches_exact_rational_arithmetic():
    """ceil(alpha K) and ceil((n + 1)(1 - alpha)) for every alpha
    = 0.01 ... 0.99 and K, n up to 400, against exact fractions. 0.8.0
    used np.ceil on the float product and was wrong in 141 (alpha K)
    and 577 ((n + 1)(1 - alpha)) cases up to 1000 (e.g. 0.07 * 100)."""
    for i in range(1, 100):
        a = float(f"{i / 100:.2f}")
        fa = Fraction(f"{a:.2f}")
        for K in range(1, 401):
            assert _ceil_count(a * K) == _exact_ceil(fa * K)
            assert _ceil_count((K + 1) * (1.0 - a)) == \
                _exact_ceil((K + 1) * (1 - fa))
    # values that are not whole numbers still round up
    assert _ceil_count(3.2) == 4 and _ceil_count(0.5) == 1
    assert _ceil_count(7.0) == 7


def test_cvar_counts_the_right_number_of_worst_values():
    v = np.arange(100.0)
    # 0.07 * 100 = 7 exactly: the mean of 0..6 is 3.0 (0.8.0: 3.5,
    # the mean of the 8 smallest)
    assert ft.cvar(v, 0.07) == np.mean(np.arange(7.0)) == 3.0
    assert ft.tail_statistics(v, 0.07)["CVaR"] == 3.0
    # the interval reports the same tail size
    rng = np.random.default_rng(0)
    J = rng.normal(size=400)
    ci = ft.cvar_interval(J, 0.07)
    assert ci["n_tail"] == 28
    assert ci["CVaR"] == np.sort(J)[:28].mean()


def test_cvar_objective_averages_exactly_alpha_K_draws():
    """The sorting estimator of `robustify` (via
    `cvar_objective_and_grad`) with alpha = 0.07, K = 100 averages the
    7 worst draws, checked against its own returned merits sorted
    here."""
    lam = np.linspace(0.45, 0.65, 21)
    S = np.ones(lam.size)
    w, c = ft.notch_weights(lam, 0.55, 0.02, 0.02)
    box = ft.DesignBox(4, 0.03, 0.15, 1.6, 2.3)
    t = np.array([0.06, 0.09, 0.07, 0.1])
    n = np.array([2.1, 1.7, 2.1, 1.7])
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 0.02, (40, 8))
    twin = ft.GaussianTwin(x)
    z = rng.normal(size=(100, 8))
    val, _, _, J = ft.cvar_objective_and_grad(lam, t, n, S, w, c, twin, z,
                                              0.07, box, n_sub=1.5)
    assert abs(val - np.sort(J)[:7].mean()) < 1e-15
    assert abs(val - np.sort(J)[:8].mean()) > 1e-6     # 0.8.0's value


def test_conformal_rank_is_the_exact_formula():
    s = np.arange(1.0, 150.0)                    # n = 149 scores 1..149
    k = _exact_ceil(150 * (1 - Fraction("0.18")))
    assert k == 123
    # 0.8.0: 124.0 and 124/150
    assert ft.conformal_quantile(s, 0.18) == 123.0
    assert ft.conformal_coverage_exact(149, 0.18) == 123 / 150
    aci = ft.AdaptiveConformal(s, alpha=0.18, window=200)
    assert aci.half_width() == 123.0


# --------------- bug 2: uncertainty with no noise estimate ---------------


def _one_layer(lam):
    t0 = np.array([0.080])
    n0 = np.array([2.0])
    S = np.ones(lam.size)
    T = ft.transmittance(lam, t0 * 1.01, n0[:, None] * S, 1.0, 1.5)
    return t0, n0, S, T


def test_zero_dof_without_noise_level_gives_nan_not_unit_noise():
    lam = np.array([0.55])
    t0, n0, S, T = _one_layer(lam)
    with pytest.warns(RuntimeWarning, match="0 degrees of freedom"):
        rec = ft.errors_from_spectrum(lam, T, t0, n0, S, n_sub=1.5)
    assert abs(rec.dt_over_t[0] - 0.01) < 1e-8 and rec.dof == 0
    assert np.isnan(rec.sigma).all()             # 0.8.0: 7.400
    m = [ft.Measurement(0.0, "s", "T", T)]
    with pytest.warns(RuntimeWarning, match="0 degrees of freedom"):
        jr = ft.errors_from_spectra(lam, m, t0, n0, S, n_sub=1.5)
    assert np.isnan(jr.sigma_t).all()
    with pytest.raises(ValueError, match="bootstrap needs a noise level"):
        ft.errors_from_spectra(lam, m, t0, n0, S, n_sub=1.5, n_boot=5)


def test_zero_dof_with_noise_level_matches_finite_difference():
    """With sigma_T given, dof = 0 is fine: sigma = sigma_T / |dT/dx|,
    the derivative taken here by central differences of the forward
    solver."""
    lam = np.array([0.55])
    t0, n0, S, T = _one_layer(lam)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rec = ft.errors_from_spectrum(lam, T, t0, n0, S, n_sub=1.5,
                                      sigma_T=0.002)
    h = 1e-6

    def Tx(x):
        return ft.transmittance(lam, t0 * (1 + x), n0[:, None] * S, 1.0,
                                1.5)[0]
    d = (Tx(0.01 + h) - Tx(0.01 - h)) / (2 * h)
    assert abs(rec.sigma[0] - 0.002 / abs(d)) < 1e-6 * rec.sigma[0]


# ------------------------- input checks -------------------------


def test_angle_and_band_width_checks():
    lam = np.array([0.5])
    with pytest.raises(ValueError, match="radians"):
        ft.stack_rt(lam, [0.1], [2.0], 1.0, 1.5, 2.0, "s")   # 0.8.0: R of pi-2
    with pytest.raises(ValueError):
        ft.transmittance(lam, [0.1], [2.0], 1.0, 1.5, np.nan, "p")
    # a negative angle is the mirror image: same R and T (unchanged)
    for pol in "spu":
        a = ft.stack_rt(lam, [0.1], [2.0], 1.0, 1.5, 0.6, pol)
        b = ft.stack_rt(lam, [0.1], [2.0], 1.0, 1.5, -0.6, pol)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    lg = np.linspace(0.5, 0.56, 7)
    with pytest.raises(ValueError, match="guard_um"):
        ft.notch_weights(lg, 0.53, 0.01, -0.02)
    with pytest.raises(ValueError, match="guard_um"):
        ft.bandpass_weights(lg, 0.52, 0.54, -0.01)
    LAM = np.linspace(0.4, 0.8, 41)
    t0, n0, S, T = _one_layer(LAM)
    with pytest.raises(ValueError, match="max_error"):
        ft.errors_from_spectrum(LAM, T, t0, n0, S, n_sub=1.5, max_error=0)


def test_trace_file_with_byte_order_mark_and_bad_field(tmp_path):
    p = tmp_path / "t.csv"
    body = ("run,layer,t_recipe_um,n_recipe,t_fab_um,n_fab\n"
            "1,1,0.05,2.0,0.051,2.01\n1,2,0.06,1.5,0.059,1.49\n")
    p.write_text(body, encoding="utf-8-sig")     # as Excel writes it
    rt, rn, ftd, fnd = ft.load_traces_csv(p)
    assert np.array_equal(rt, [[0.05, 0.06]])
    assert np.array_equal(fnd, [[2.01, 1.49]])
    p.write_text(body.replace("0.059", "n/a"))
    with pytest.raises(ValueError, match="line 3"):
        ft.load_traces_csv(p)


def test_version_is_the_same_everywhere():
    if not (ROOT / "pyproject.toml").exists():   # pragma: no cover
        pytest.skip("not run from the repository")
    py = (ROOT / "pyproject.toml").read_text()
    cff = (ROOT / "CITATION.cff").read_text()
    assert re.search(r'^version = "(.+)"', py, re.M).group(1) == \
        ft.__version__
    assert re.search(r"^version: (.+)$", cff, re.M).group(1).strip() == \
        ft.__version__


# --------------------- feature: load_spectrum_csv ---------------------


def _cary_like(lam_nm, pct_T, pct_R):
    """A file laid out like a typical spectrophotometer CSV export:
    two header lines, a scan from long to short wavelengths, a
    trailing comma on every row, then a metadata block."""
    out = ["Sample 17,,,", "Wavelength (nm),%T,%R,"]
    for a, b, c in zip(lam_nm[::-1], pct_T[::-1], pct_R[::-1]):
        out.append(f"{float(a)!r},{float(b)!r},{float(c)!r},")
    out += ["", "Collection Time: 10/10/2026 10:15:00", "Scan rate,600",
            "Instrument,demo"]
    return "\n".join(out) + "\n"


def test_reader_converts_units_and_order(tmp_path):
    lam_nm = np.linspace(400.0, 800.0, 81)
    pT = 50.0 + 40.0 * np.sin(lam_nm / 37.0)
    pR = 100.0 - pT
    p = tmp_path / "scan.csv"
    p.write_text(_cary_like(lam_nm, pT, pR))
    lam, T = ft.load_spectrum_csv(p, "nm", "percent")
    assert np.array_equal(lam, lam_nm * 1e-3)    # increasing, in um
    assert np.array_equal(T, pT * 1e-2)          # fractions
    lam2, R = ft.load_spectrum_csv(p, "nm", "percent", column=2)
    assert np.array_equal(lam2, lam) and np.array_equal(R, pR * 1e-2)
    # tab- and blank-separated files, um and fractions, a BOM
    for sep in ("\t", "   "):
        q = tmp_path / "scan.txt"
        rows = "\n".join(f"{float(a)!r}{sep}{float(b)!r}" for a, b in
                         zip(lam_nm * 1e-3, pT * 1e-2))
        q.write_text("lambda (um)" + sep + "T\n" + rows + "\n",
                     encoding="utf-8-sig")
        lam3, T3 = ft.load_spectrum_csv(q, "um", "fraction")
        assert np.array_equal(lam3, lam_nm * 1e-3)
        assert np.array_equal(T3, pT * 1e-2)
    # semicolon-separated (decimal points)
    rows = "\n".join(f"{float(a)!r};{float(b)!r}" for a, b in
                     zip(lam_nm, pT))
    q.write_text("nm;%T\n" + rows + "\n")
    lam4, T4 = ft.load_spectrum_csv(q, "nm", "percent")
    assert np.array_equal(lam4, lam) and np.array_equal(T4, T)


@pytest.mark.parametrize("text, kw, msg", [
    ("nm,%T\n400,50\n400,51\n", {}, "line 3"),          # repeat
    ("nm,%T\n400,50\n410,51\n405,52\n", {}, "line 4"),  # not monotonic
    ("nm,%T\n400,50,1\n410,51\n", {}, "line 3"),        # ragged
    ("nm,%T\n400,50\n410,51\nbad row\n420,52\n", {}, "damaged"),
    ("nm;%T\n400,0;50,5\n410,0;51,5\n", {}, "decimal commas"),
    ("nm,%T\n400,50\n410,nan\n", {}, "non-finite"),
    ("nm,%T\n400,50\n410,51\n", {"column": 2}, "does not exist"),
    ("only words here\n", {}, "no rows of numbers"),
    ("nm,T\n400,50\n410,51\n", {"value_unit": "fraction"}, "percent"),
    ("um,T\n400,0.5\n410,0.51\n", {"wavelength_unit": "um"},
     "nanometers"),
    ("nm,T\n0.4,50\n0.41,51\n", {}, "micrometers"),
    ("nm,T\n400,50\n410,51\n", {"wavelength_unit": "A"}, "'nm' or 'um'"),
])
def test_reader_refuses_what_it_cannot_read_reliably(tmp_path, text, kw,
                                                     msg):
    p = tmp_path / "bad.csv"
    p.write_text(text)
    args = dict(wavelength_unit="nm", value_unit="percent")
    args.update(kw)
    with pytest.raises(ValueError, match=msg):
        ft.load_spectrum_csv(p, **args)


def test_reader_to_recovery_end_to_end(tmp_path):
    """A coated plate measured in nm and percent, written in an
    instrument-like layout, read back and fitted: the thickness errors
    come back to 1e-6 (noise-free)."""
    lam_nm = np.linspace(400.0, 800.0, 121)
    lam = lam_nm * 1e-3
    S = ft.dispersion_shape(lam, 0.55)
    nsub = ft.SIO2_MALITSON1965.n(lam)
    t0 = np.array([0.070, 0.095, 0.060, 0.110, 0.080])
    n0 = np.array([2.2, 1.5, 2.2, 1.5, 2.2])
    xt = np.array([0.03, -0.02, 0.015, 0.01, -0.025])
    R, T = ft.stack_rt(lam, t0 * (1 + xt), n0[:, None] * S, 1.0, nsub,
                       n_exit=1.0)
    p = tmp_path / "plate.csv"
    p.write_text(_cary_like(lam_nm, 100 * T, 100 * R))
    lam_r, T_r = ft.load_spectrum_csv(p, "nm", "percent")
    assert np.max(np.abs(lam_r - lam)) < 1e-15
    rec = ft.errors_from_spectrum(lam_r, T_r, t0, n0, S, n_sub=nsub,
                                  n_exit=1.0)
    assert np.max(np.abs(rec.dt_over_t - xt)) < 1e-6


# -------------------- feature: recovery_precision --------------------


def _fd_jacobian(lam, meas, t0, n0, S, mask, n_sub, n_exit, h=1e-6):
    """Weighted Jacobian by central differences of fabtwin.tmm.stack_rt
    (the forward solver; no adjoint involved)."""
    N = t0.size
    idx = np.nonzero(mask)[0]

    def spectra(p):
        t = t0 * (1 + p[:N])
        n = n0.copy()
        n[idx] += p[N:]
        out = []
        for m in meas:
            R, T = ft.stack_rt(lam, t, n[:, None] * S, 1.0, n_sub,
                               m.theta0_rad, m.pol, n_exit=n_exit)
            out.append((T if m.quantity == "T" else R) / m.sigma)
        return np.concatenate(out)
    P = N + idx.size
    cols = []
    for j in range(P):
        e = np.zeros(P)
        e[j] = h
        cols.append((spectra(e) - spectra(-e)) / (2 * h))
    return np.array(cols).T


def test_planner_equals_independent_finite_difference_fisher():
    lam = np.linspace(0.42, 0.86, 45)
    S = ft.dispersion_shape(lam, 0.55)
    nsub = ft.SIO2_MALITSON1965.n(lam)
    t0 = np.array([0.085, 0.120, 0.070])
    n0 = np.array([2.1, 1.46, 2.1])
    meas = [ft.Measurement(0.0, "s", "T", None, 0.002),
            ft.Measurement(np.deg2rad(50), "p", "T", None, 0.003),
            ft.Measurement(np.deg2rad(50), "s", "R", None, 0.003)]
    mask = np.array([True, False, True])
    for n_exit in (None, 1.0):
        plan = ft.recovery_precision(lam, meas, t0, n0, S, fit_index=mask,
                                     n_sub=nsub, n_exit=n_exit)
        J = _fd_jacobian(lam, meas, t0, n0, S, mask, nsub, n_exit)
        C = np.linalg.inv(J.T @ J)
        assert np.allclose(plan.covariance, C, rtol=1e-5, atol=0)
        assert np.allclose(plan.sigma_t, np.sqrt(np.diag(C))[:3],
                           rtol=1e-5)
        assert plan.sigma_n[1] == 0.0
        assert np.allclose(plan.sigma_n[[0, 2]], np.sqrt(np.diag(C))[3:],
                           rtol=1e-5)
        assert plan.n_values == 3 * lam.size and plan.n_unknowns == 5
        assert np.allclose(np.diag(plan.correlation), 1.0)


def test_planner_one_layer_closed_form():
    """One layer, thickness only: sigma = sigma_T / sqrt(sum_l
    (dT_l/dx)^2), the derivative by central differences."""
    lam = np.linspace(0.45, 0.75, 31)
    S = np.ones(lam.size)
    t0, n0 = np.array([0.11]), np.array([2.05])
    plan = ft.recovery_precision(
        lam, [ft.Measurement(0.0, "s", "T", None, 0.004)], t0, n0, S,
        n_sub=1.52)
    h = 1e-6

    def Tx(x):
        return ft.transmittance(lam, t0 * (1 + x), n0[:, None] * S, 1.0,
                                1.52)
    d = (Tx(h) - Tx(-h)) / (2 * h)
    assert abs(plan.sigma_t[0] - 0.004 / np.sqrt(np.sum(d * d))) \
        < 1e-6 * plan.sigma_t[0]


def test_planner_predicts_the_scatter_of_actual_fits():
    """200 simulated noisy spectra of one stack (0.1 % noise), each
    fitted by errors_from_spectrum: the scatter of the recovered errors
    matches the predicted sigma (the relative standard error of a
    standard deviation from 200 draws is about 5 %; tolerance 20 %;
    observed ratios 1.07, 1.04, 0.97), and the planner agrees with the
    sigma the fits report. (At 0.3 % noise the same stack's errors are
    determined only to about 1 %, the spectrum is no longer linear
    over that range, and the observed scatter was 17 % to 33 % larger
    than predicted: the prediction is a first-order one, as its
    docstring says.)"""
    lam = np.linspace(0.40, 0.80, 81)
    S = ft.dispersion_shape(lam, 0.55)
    nsub = ft.SIO2_MALITSON1965.n(lam)
    t0 = np.array([0.075, 0.105, 0.065])
    n0 = np.array([2.0, 1.46, 2.0])
    xt = np.array([0.02, -0.015, 0.01])
    sig = 0.001
    plan = ft.recovery_precision(
        lam, [ft.Measurement(0.0, "s", "T", None, sig)], t0, n0, S,
        n_sub=nsub, at_dt_over_t=xt)
    T_true = ft.transmittance(lam, t0 * (1 + xt), n0[:, None] * S, 1.0,
                              nsub)
    rng = np.random.default_rng(11)
    got, reported = [], []
    for _ in range(200):
        Tm = T_true + rng.normal(0.0, sig, lam.size)
        assert 0 <= Tm.min() and Tm.max() <= 1
        rec = ft.errors_from_spectrum(lam, Tm, t0, n0, S, n_sub=nsub,
                                      sigma_T=sig, n_starts=1)
        got.append(rec.dt_over_t)
        reported.append(rec.sigma)
    got = np.array(got)
    scatter = got.std(axis=0, ddof=1)
    assert np.all(np.abs(scatter / plan.sigma_t - 1) < 0.20)
    assert np.all(np.abs(got.mean(axis=0) - xt) < 4 * plan.sigma_t
                  / np.sqrt(200))
    assert np.allclose(np.mean(reported, axis=0), plan.sigma_t, rtol=0.02)


def test_planner_agrees_with_joint_recovery_at_the_recipe():
    """errors_from_spectra on noise-free spectra of the recipe itself
    converges to zero errors; the sigma it reports there is the same
    quantity the planner predicts."""
    lam = np.linspace(0.40, 0.90, 61)
    S = ft.dispersion_shape(lam, 0.55)
    t0 = np.array([0.070, 0.095, 0.060])
    n0 = np.array([2.2, 1.5, 2.2])
    plan_m, meas = [], []
    for a, p in ((0, "s"), (45, "s"), (45, "p"), (60, "p")):
        g = ft.stack_rta_and_grads(lam, t0, n0[:, None] * S, 1.0, 1.46,
                                   np.deg2rad(a), p)
        meas.append(ft.Measurement(np.deg2rad(a), p, "T", g["T"], 0.002))
        plan_m.append(ft.Measurement(np.deg2rad(a), p, "T", None, 0.002))
    r = ft.errors_from_spectra(lam, meas, t0, n0, S, fit_index=True,
                               n_sub=1.46, n_starts=2)
    plan = ft.recovery_precision(lam, plan_m, t0, n0, S, fit_index=True,
                                 n_sub=1.46)
    assert np.max(np.abs(r.dt_over_t)) < 1e-8
    assert np.allclose(plan.sigma_t, r.sigma_t, rtol=1e-6)
    assert np.allclose(plan.sigma_n, r.sigma_n, rtol=1e-6)


def test_planner_refusals():
    lam = np.linspace(0.40, 0.80, 41)
    S = ft.dispersion_shape(lam, 0.55)
    m = ft.Measurement(0.0, "s", "T", None, 0.002)
    t0 = np.array([0.08, 0.06, 0.10])
    with pytest.raises(ValueError, match="noise level"):
        ft.recovery_precision(lam, [ft.Measurement(0.0, "s", "T", None)],
                              t0, np.array([2.0, 1.5, 2.0]), S, n_sub=1.46)
    with pytest.raises(ValueError, match="underdetermined"):
        ft.recovery_precision(lam[:2], [m], t0, np.array([2.0, 1.5, 2.0]),
                              S[:2], n_sub=1.46)
    # adjacent layers of the same index: only their sum is visible, and
    # the message names both layers
    with pytest.raises(ValueError, match="layer 1 thickness.*layer 2 "
                                         "thickness|layer 2 thickness.*"
                                         "layer 1 thickness"):
        ft.recovery_precision(lam, [m], t0, np.array([2.0, 2.0, 1.46]), S,
                              n_sub=1.46)
    with pytest.raises(ValueError):
        ft.recovery_precision(lam, [m], t0, np.array([2.0, 1.5, 2.0]), S,
                              n_sub=1.46, at_dt_over_t=[0.1, 0.0])
