"""0.10.0: two bug fixes (a conformal rank of 0; NaN for opaque
layers), the batched adjoint behind `robustify`, and the optical
monitoring planner `monitoring_plan`.

Independent references used here: exact rational arithmetic
(`fractions.Fraction`); the Fresnel reflectance of a semi-infinite
absorbing medium and its derivatives, written out by hand; total
internal reflection; the independent `tmm` package; the unchanged
public `stack_BC` (the 0.9.1 arithmetic) and the per-draw
`merit_and_grad` loop of 0.9.1, kept here as references; the closed
form of one lossless layer on a substrate; partial stacks computed
separately by `stack_rt`; central finite differences of `stack_rt`;
and simulated monitor stops found by root finding on `stack_rt`."""
import warnings
from fractions import Fraction

import numpy as np
import pytest
from scipy.optimize import brentq

import fabtwin as ft
from fabtwin import adjoint, robust
from fabtwin.risk import _ceil_count
from fabtwin.tmm import stack_BC


# --------------- bug 1: a conformal rank of 0 (largest score) ---------------


def test_positive_count_is_never_zero():
    """0.9.1: _ceil_count(x) was 0 for 0 < x <= 1e-9."""
    assert _ceil_count(1e-12) == 1 and _ceil_count(1e-10) == 1
    assert _ceil_count(0.0) == 0
    for k in range(1, 16):
        a = 1.0 - 10.0 ** -k
        for n in (1, 5, 10, 100, 1000):
            exact = Fraction(n + 1) * (1 - Fraction(a))
            want = -(-exact.numerator // exact.denominator)
            assert _ceil_count((n + 1) * (1.0 - a)) == want


def test_conformal_quantile_near_alpha_one_is_the_smallest_score():
    s = np.arange(1.0, 11.0)
    # rank ceil(11 * 1e-12) = 1: the smallest score (0.9.1: 10.0, the
    # largest, through index -1) and coverage 1/11 (0.9.1: 0.0)
    assert ft.conformal_quantile(s, 1 - 1e-12) == 1.0
    assert ft.conformal_coverage_exact(10, 1 - 1e-12) == 1 / 11
    assert ft.mondrian_quantiles(s, ["a"] * 10, 1 - 1e-12) == {"a": 1.0}
    # unchanged where the rank was already right
    assert ft.conformal_quantile(s, 0.5) == 6.0


def test_adaptive_conformal_half_width_just_below_alpha_one():
    """A natural trajectory: 60 runs with score 0 drive alpha_t to
    0.9999999999999984 (1 in exact arithmetic). The half-width must be
    the smallest score (rank 1), not the largest (0.9.1: 2.3250)."""
    rng = np.random.default_rng(0)
    aci = ft.AdaptiveConformal(np.abs(rng.normal(size=100)), alpha=0.25,
                               gamma=0.05)
    for _ in range(60):
        aci.update(0.0)
    assert 0.0 < 1.0 - aci.alpha_t < 1e-9
    n = len(aci.scores)
    a = Fraction(aci.alpha_t)
    rank = -(-((n + 1) * (1 - a)).numerator // ((n + 1) * (1 - a)).denominator)
    assert rank == 1
    assert aci.half_width() == min(aci.scores) == 0.0
    assert max(aci.scores) > 2.3


# ----------------- bug 2: NaN for opaque (thick absorbing) layers -----------------


def _fresnel_R(N):
    """Reflectance of a semi-infinite medium of index N from air, and
    its derivatives with respect to n and k (N = n + i k), by hand."""
    n, k = N.real, N.imag
    num = (1 - n) ** 2 + k ** 2
    den = (1 + n) ** 2 + k ** 2
    dn = (-2 * (1 - n) * den - num * 2 * (1 + n)) / den ** 2
    dk = (2 * k * den - num * 2 * k) / den ** 2
    return num / den, dn, dk


@pytest.mark.parametrize("t_um", [10.0, 20.0, 200.0])
def test_opaque_metal_gives_the_fresnel_closed_form(t_um):
    """0.9.1 returned NaN for R and T (10 um of n = 1 + 6i at 0.5 um:
    |Im delta| = 754); the metal is opaque, so R is the Fresnel
    reflectance of the metal alone, 36/40 = 0.9 for N = 1 + 6i."""
    N = 1.0 + 6.0j
    lam = np.array([0.5])
    for pol in ("s", "p", "u"):
        R, T = ft.stack_rt(lam, [t_um], np.array([N]), 1.0, 1.52, 0.0, pol)
        assert abs(R[0] - 0.9) < 1e-12 and T[0] == 0.0
    g = ft.stack_rta_and_grads(lam, [t_um], np.array([N]), 1.0, 1.52)
    R0, dn, dk = _fresnel_R(N)
    assert abs(g["R"][0] - R0) < 1e-12 and g["T"][0] == 0.0
    assert abs(g["A"][0] - (1 - R0)) < 1e-12
    assert abs(g["dR_dn"][0, 0] - dn) < 1e-12
    assert abs(g["dR_dk"][0, 0] - dk) < 1e-12
    assert abs(g["dR_dt"][0, 0]) < 1e-12 and g["dT_dt"][0, 0] == 0.0


def test_opaque_layer_under_a_coating_matches_tmm_package():
    byrnes = pytest.importorskip("tmm")
    lam = np.array([0.5, 0.6])
    t = [0.1, 10.0]
    n = np.array([1.46, 1.0 + 6.0j])
    for th, pol in ((0.0, "s"), (0.6, "s"), (0.6, "p")):
        R, T = ft.stack_rt(lam, t, n, 1.0, 1.52, th, pol)
        g = ft.stack_rta_and_grads(lam, t, n, 1.0, 1.52, th, pol)
        for i, L in enumerate(lam):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ref = byrnes.coh_tmm(pol, [1.0, 1.46, 1.0 + 6.0j, 1.52],
                                     [np.inf, 100.0, 1e4, np.inf], th,
                                     L * 1000)
            assert abs(R[i] - ref["R"]) < 1e-12
            assert abs(g["R"][i] - ref["R"]) < 1e-12
            # tmm caps opaque layers at 1e-30 transmission on purpose
            assert T[i] < 1e-300 and ref["T"] < 1e-29
    # the derivative of R with respect to the top layer still matches a
    # central finite difference of the forward optics
    h = 1e-6
    Rp = ft.stack_rt(lam, [0.1 + h, 10.0], n, 1.0, 1.52)[0]
    Rm = ft.stack_rt(lam, [0.1 - h, 10.0], n, 1.0, 1.52)[0]
    g = ft.stack_rta_and_grads(lam, t, n, 1.0, 1.52)
    np.testing.assert_allclose(g["dR_dt"][0], (Rp - Rm) / (2 * h),
                               rtol=1e-7, atol=1e-9)


def test_thick_evanescent_gap_reflects_totally():
    """Glass (1.5) -> 70 um of air -> glass at 60 degrees: beyond the
    critical angle the gap is opaque (|Im delta| about 730 at 0.5 um),
    so R = 1 and T = 0 (0.9.1: NaN)."""
    lam = np.array([0.5])
    for pol in ("s", "p"):
        R, T = ft.stack_rt(lam, [70.0], np.array([1.0]), 1.5, 1.5,
                           np.deg2rad(60), pol)
        assert abs(R[0] - 1.0) < 1e-12 and T[0] == 0.0
        g = ft.stack_rta_and_grads(lam, [70.0], np.array([1.0]), 1.5, 1.5,
                                   np.deg2rad(60), pol)
        assert abs(g["R"][0] - 1.0) < 1e-12 and np.isfinite(g["dR_dn"]).all()


def _old_rt(lam, t, n, n_inc, n_sub):
    """The 0.9.1 arithmetic of stack_rt (through the unchanged stack_BC)."""
    B, C, eta0, eta_sub = stack_BC(lam, t, n, n_inc, n_sub)
    D = eta0 * B + C
    R = np.abs((eta0 * B - C) / D) ** 2
    T = 4.0 * np.real(eta0) * np.real(eta_sub) / np.abs(D) ** 2
    return R, T


def test_opaque_rescaling_leaves_other_results_unchanged():
    lam = np.array([0.5])
    n = np.array([1.46, 1.0 + 6.0j])
    # sum |Im delta| = 264, below the threshold: the old path, bit for bit
    R, T = ft.stack_rt(lam, [0.1, 3.5], n, 1.0, 1.52)
    Ro, To = _old_rt(lam, [0.1, 3.5], n, 1.0, 1.52)
    assert np.array_equal(R, Ro) and np.array_equal(T, To)
    # 339: rescaled; 0.9.1 was still finite here and agrees to round-off
    R, T = ft.stack_rt(lam, [0.1, 4.5], n, 1.0, 1.52)
    Ro, To = _old_rt(lam, [0.1, 4.5], n, 1.0, 1.52)
    assert To[0] > 0
    np.testing.assert_allclose(R, Ro, rtol=1e-13)
    np.testing.assert_allclose(T, To, rtol=1e-11)


# ------------------ speed: all draws through the adjoint at once ------------------


def _draws_reference(lam, t, n0, S, w, const, twin, z, box, n_inc, n_sub):
    """The per-draw loop of 0.9.1 (fabtwin.robust._draws, model=None)."""
    N = t.size
    x = twin.transform(z)
    t_lo, t_hi, n_lo, n_hi = robust._fab_clip_bounds(box)
    J = np.empty(len(x))
    gT = np.empty((len(x), N))
    gN = np.empty((len(x), N))
    for k in range(len(x)):
        tk_raw = t * (1.0 + x[k, :N])
        nk_raw = n0 + x[k, N:]
        tk = np.clip(tk_raw, t_lo, t_hi)
        nk = np.clip(nk_raw, n_lo, n_hi)
        Jk, gdk, gnk = ft.merit_and_grad(lam, tk, nk, S, w, const,
                                         n_inc=n_inc, n_sub=n_sub)
        J[k] = Jk
        gT[k] = gdk * (1.0 + x[k, :N]) * ((tk_raw > t_lo) & (tk_raw < t_hi))
        gN[k] = gnk * ((nk_raw > n_lo) & (nk_raw < n_hi))
    return J, gT, gN


def _equal_to_rounding(a, b, rel=1e-12):
    """a equals b up to rel times the largest |b|.

    0.10.0 asked for bit-for-bit equality here. That held on the
    development machine, but it failed once on a GitHub test machine
    (oldest-dependencies job, NumPy 1.26.0) and passed for the same code
    in two later runs: the last digits of the batched arithmetic can
    differ between machines. 1e-12 is about 4500 times the double
    precision step and far below any change a code error would cause.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape
    scale = float(np.max(np.abs(b))) if b.size else 0.0
    assert float(np.max(np.abs(a - b), initial=0.0)) <= rel * scale


def test_batched_draws_equal_the_per_draw_loop():
    rng = np.random.default_rng(1)
    for case in range(25):
        N = int(rng.integers(1, 16))
        L = int(rng.integers(1, 90))
        lam = np.sort(rng.uniform(0.4, 0.9, L))
        S = rng.uniform(0.95, 1.05, L) if case % 3 else \
            rng.uniform(0.95, 1.05, (N, L))
        w = rng.normal(size=L)
        box = ft.DesignBox(N, 0.02, 0.15, 1.5, 2.3)
        t = rng.uniform(0.02, 0.15, N)
        n0 = rng.uniform(1.5, 2.3, N)
        twin = ft.GaussianTwin(rng.normal(0, 0.08, (200, 2 * N)))
        z = rng.normal(size=(int(rng.integers(1, 300)), 2 * N))
        nsub = rng.uniform(1.4, 1.6, L) if case % 2 else 1.52
        new = robust._draws(lam, t, n0, S, w, 0.3, twin, z, box, 1.0, nsub,
                            None)
        ref = _draws_reference(lam, t, n0, S, w, 0.3, twin, z, box, 1.0,
                               nsub)
        for a, b in zip(new, ref):
            _equal_to_rounding(a, b)


def test_batched_adjoint_rows_equal_merit_and_grad():
    rng = np.random.default_rng(2)
    lam = np.linspace(0.45, 0.65, 33)
    S = ft.dispersion_shape(lam, 0.55)
    w, c = ft.notch_weights(lam, 0.532, 0.012, 0.02)
    t = rng.uniform(0.03, 0.15, (7, 6))
    n = rng.uniform(1.6, 2.3, (7, 6))
    J, gd, gn = adjoint._merits_and_grads_batch(lam, t, n, S, w, c,
                                                n_sub=1.46)
    for k in range(7):
        Jk, gdk, gnk = ft.merit_and_grad(lam, t[k], n[k], S, w, c,
                                         n_sub=1.46)
        _equal_to_rounding(J[k], Jk)
        _equal_to_rounding(gd[k], gdk)
        _equal_to_rounding(gn[k], gnk)
    with pytest.raises(ValueError, match="lossless"):
        adjoint._merits_and_grads_batch(lam, t, n, S, w, c, n_sub=1.46 + 0j)


# ----------------------- new: the optical monitoring planner -----------------------


def _one_layer_T(delta, n0, n1, ns):
    """Closed form: one lossless layer on a substrate, normal incidence,
    from the characteristic matrix: eta0 B + C = (n0 + ns) cos d
    + i (n0 ns / n1 + n1) sin d."""
    a = n0 + ns
    b = n0 * ns / n1 + n1
    F = a ** 2 * np.cos(delta) ** 2 + b ** 2 * np.sin(delta) ** 2
    dF = (b ** 2 - a ** 2) * np.sin(2 * delta)
    return 4 * n0 * ns / F, -4 * n0 * ns * dF / F ** 2


@pytest.mark.parametrize("n1,ns,lam,qw", [(2.1, 1.52, 0.55, 2.3),
                                          (1.38, 1.9, 0.633, 0.7),
                                          (2.3, 1.46, 0.8, 1.2)])
def test_one_layer_curve_slope_and_turning_points_match_closed_form(
        n1, ns, lam, qw):
    t1 = qw * lam / (4 * n1)
    p = ft.monitoring_plan(lam, [t1], [n1], n_sub=ns)
    grid, vals = p.curves[0]
    delta = 2 * np.pi * n1 * grid / lam
    T_cf, dT_cf = _one_layer_T(delta, 1.0, n1, ns)
    np.testing.assert_allclose(vals, T_cf, rtol=0, atol=1e-13)
    slope_cf = dT_cf[-1] * 2 * np.pi * n1 / lam
    assert abs(p.slope[0] - slope_cf) < 1e-10 * max(1.0, abs(slope_cf))
    # turning points at delta = m pi / 2 inside the layer
    assert p.turning_points[0] == int(np.floor(qw))
    assert abs(p.start[0] - 4 * ns / (1 + ns) ** 2) < 1e-14


def test_cutoffs_are_the_partial_stacks_and_slopes_finite_differences():
    lam = 0.6
    t = np.array([0.071, 0.093, 0.064, 0.115, 0.082])
    n = np.array([2.2, 1.46, 2.2 + 0.01j, 1.46, 2.2])
    for th, pol, q, n_exit in ((0.0, "s", "T", None), (0.2, "p", "R", None),
                               (0.0, "s", "T", 1.0)):
        p = ft.monitoring_plan(lam, t, n, n_sub=1.52, quantity=q,
                               theta0_rad=th, pol=pol, n_exit=n_exit)
        assert list(p.order) == [4, 3, 2, 1, 0]
        k = 0 if q == "R" else 1
        for j in range(5):
            part = ft.stack_rt([lam], t[j:], n[j:], 1.0, 1.52, th, pol,
                               n_exit)[k][0]
            assert abs(p.cutoff[j] - part) < 1e-13
            if j < 4:
                assert abs(p.start[j] - p.cutoff[j + 1]) < 1e-13
            h = 1e-6
            tp = t[j:].copy()
            tm = t[j:].copy()
            tp[0] += h
            tm[0] -= h
            fd = (ft.stack_rt([lam], tp, n[j:], 1.0, 1.52, th, pol,
                              n_exit)[k][0]
                  - ft.stack_rt([lam], tm, n[j:], 1.0, 1.52, th, pol,
                                n_exit)[k][0]) / (2 * h)
            assert abs(p.slope[j] - fd) < 1e-7 * max(1.0, abs(fd))


def test_quarter_wave_stack_at_its_design_wavelength_is_refused():
    lam0 = 0.55
    n = np.array([2.1, 1.46] * 3)
    t = lam0 / (4 * n)
    with pytest.warns(RuntimeWarning, match="turning point"):
        p = ft.monitoring_plan(lam0, t, n, n_sub=1.52, sigma_signal=1e-3)
    assert np.all(np.isnan(p.sigma_t)) and np.isnan(p.worst_rel_sigma_t())
    assert np.all(p.slope_ratio < 1e-12)
    assert np.all(p.turning_points == 0)
    # half-wave layers pass exactly one turning point inside
    with pytest.warns(RuntimeWarning):
        p2 = ft.monitoring_plan(lam0, 2 * t, n, n_sub=1.52,
                                sigma_signal=1e-3)
    assert np.all(p2.turning_points == 1)
    # away from the design wavelength the stops are well defined
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p3 = ft.monitoring_plan(0.62, t[:5], n[:5], n_sub=1.52,
                                sigma_signal=1e-3)
    assert np.all(np.isfinite(p3.sigma_t))


def test_predicted_thickness_error_matches_simulated_stops():
    """sigma_t = sigma / |slope| against stops simulated with the
    forward optics only: the layer is stopped where the noisy signal
    first reaches the cut-off, signal(t) = cutoff - noise, solved by
    root finding on stack_rt."""
    lam = 0.52
    t = np.array([0.075, 0.098, 0.066, 0.118])
    n = np.array([2.1, 1.46, 2.1, 1.46])
    sig = 5e-4
    p = ft.monitoring_plan(lam, t, n, n_sub=1.52, sigma_signal=sig)
    assert np.all(p.nonlinearity < 0.05)
    rng = np.random.default_rng(4)
    for j in range(4):
        def f(tau, target):
            return ft.stack_rt([lam], np.concatenate([[tau], t[j + 1:]]),
                               n[j:], 1.0, 1.52)[1][0] - target
        st = p.sigma_t[j]
        stops = []
        for e in rng.normal(0.0, sig, 400):
            stops.append(brentq(f, t[j] - 6 * st, t[j] + 6 * st,
                                 args=(p.cutoff[j] - e,), xtol=1e-12))
        sd = np.std(np.array(stops) - t[j], ddof=1)
        assert abs(sd / st - 1) < 0.12
        assert abs(p.rel_sigma_t[j] - st / t[j]) < 1e-15


def test_monitoring_plan_refusals():
    t, n = [0.08, 0.1], [2.1, 1.46]
    for kw, msg in ((dict(lam_mon_um=0.0), "wavelength"),
                    (dict(recipe_t_um=[0.08, -0.1]), "positive"),
                    (dict(n_layers=[2.1]), "one length"),
                    (dict(quantity="A"), "quantity"),
                    (dict(sigma_signal=0.0), "sigma_signal"),
                    (dict(points_per_quarter_wave=4), "at least 8"),
                    (dict(n_layers=[2.1, -1.0]), "positive real")):
        args = dict(lam_mon_um=0.6, recipe_t_um=t, n_layers=n)
        args.update(kw)
        with pytest.raises(ValueError, match=msg):
            ft.monitoring_plan(**args)
