"""How sure is a yield or a CVaR from K draws? (0.8.0)"""
import math

import numpy as np
import pytest
from scipy.stats import norm

import fabtwin as ft


def _binom_tail_ge(k, n, p):
    return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j)
               for j in range(k, n + 1))


def _binom_tail_le(k, n, p):
    return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j)
               for j in range(0, k + 1))


def test_yield_interval_is_the_clopper_pearson_interval():
    """The bounds solve the binomial tail equations, summed directly."""
    for k, n in ((1, 10), (7, 20), (37, 40), (250, 500), (3, 3)):
        for conf in (0.9, 0.95):
            r = ft.yield_interval(k, conf, n=n)
            a = 1 - conf
            if k > 0:
                assert abs(_binom_tail_ge(k, n, r["lo"]) - a / 2) < 1e-10
            if k < n:
                assert abs(_binom_tail_le(k, n, r["hi"]) - a / 2) < 1e-10
            assert r["lo"] <= k / n <= r["hi"] and r["yield"] == k / n
    # the ends have closed forms
    for n in (5, 30, 200):
        assert abs(ft.yield_interval(0, 0.95, n=n)["hi"]
                   - (1 - 0.025 ** (1 / n))) < 1e-12
        assert abs(ft.yield_interval(n, 0.95, n=n)["lo"]
                   - 0.025 ** (1 / n)) < 1e-12
        assert ft.yield_interval(0, 0.95, n=n)["lo"] == 0.0
        assert ft.yield_interval(n, 0.95, n=n)["hi"] == 1.0
    # a boolean array is the same as its count
    ok = np.random.default_rng(0).random(57) < 0.8
    a, b = ft.yield_interval(ok), ft.yield_interval(int(ok.sum()), n=57)
    assert a == b


def test_yield_interval_covers_at_least_the_level():
    """Exact coverage by summing the binomial distribution: for every
    true yield on a grid, P(lo <= p <= hi) >= confidence."""
    n = 25
    bounds = [ft.yield_interval(k, 0.9, n=n) for k in range(n + 1)]
    worst = 1.0
    for p in np.linspace(0.005, 0.995, 199):
        cov = sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k)
                  for k in range(n + 1)
                  if bounds[k]["lo"] <= p <= bounds[k]["hi"])
        worst = min(worst, cov)
    assert worst >= 0.9 - 1e-12


def test_yield_interval_refusals():
    with pytest.raises(ValueError):
        ft.yield_interval(np.array([1.0, 0.0]))           # not boolean
    with pytest.raises(ValueError):
        ft.yield_interval(5, n=4)
    with pytest.raises(ValueError):
        ft.yield_interval(2.5, n=4)
    with pytest.raises(ValueError):
        ft.yield_interval(np.array([True]), confidence=1.0)


ALPHA = 0.1
TAU = norm.ppf(ALPHA)
CVAR_N01 = -norm.pdf(TAU) / ALPHA          # lower-tail CVaR of N(0, 1)
# E[(tau - X)_+] and E[(tau - X)_+^2] for X ~ N(0, 1), in closed form
_E1 = TAU * ALPHA + norm.pdf(TAU)
_E2 = (TAU ** 2 + 1) * ALPHA + TAU * norm.pdf(TAU)
SD_N01 = math.sqrt(_E2 - _E1 ** 2) / ALPHA  # sqrt(K) x standard error


def test_cvar_standard_error_matches_the_normal_closed_form():
    v = np.random.default_rng(0).normal(size=200_000)
    r = ft.cvar_interval(v, ALPHA)
    assert abs(r["se"] * math.sqrt(v.size) / SD_N01 - 1) < 0.02
    assert abs(r["CVaR"] - CVAR_N01) < 4 * r["se"]
    # when alpha K is whole, the linearization reproduces the estimate
    w = np.random.default_rng(1).normal(size=500)
    tau = np.sort(w)[49]
    psi = tau - np.maximum(tau - w, 0) / ALPHA
    assert abs(psi.mean() - ft.cvar(w, ALPHA)) < 1e-12


def test_cvar_interval_by_simulation():
    """2000 seeded samples of K = 400 normal draws (40 in the tail):
    the reported standard error matches the scatter of the estimates,
    and the 95 % interval covers the true CVaR at close to its level."""
    rng = np.random.default_rng(1)
    est, se, cov = [], [], 0
    for _ in range(2000):
        r = ft.cvar_interval(rng.normal(size=400), ALPHA)
        est.append(r["CVaR"])
        se.append(r["se"])
        cov += r["lo"] <= CVAR_N01 <= r["hi"]
    assert abs(np.mean(se) / np.std(est) - 1) < 0.06
    assert 0.91 <= cov / 2000 <= 0.97                  # observed 0.9315
    with pytest.raises(ValueError, match="min_tail"):
        ft.cvar_interval(rng.normal(size=150), ALPHA)    # 15 in the tail


def test_cvar_interval_with_the_fewest_tail_draws_allowed():
    """K = 200 (20 in the tail, the default min_tail): the interval
    under-covers a little, as the docs state; 2000 seeded samples."""
    rng = np.random.default_rng(5)
    est, se, cov = [], [], 0
    for _ in range(2000):
        r = ft.cvar_interval(rng.normal(size=200), ALPHA)
        est.append(r["CVaR"])
        se.append(r["se"])
        cov += r["lo"] <= CVAR_N01 <= r["hi"]
    assert abs(np.mean(se) / np.std(est) - 1) < 0.10
    assert 0.88 <= cov / 2000 <= 0.96                  # observed 0.911


def test_cvar_difference_paired_and_unpaired():
    """Common random numbers: two correlated merits. The paired
    standard error matches the scatter of the estimated difference and
    is far smaller than the unpaired one; the unpaired one is the
    root-sum-square of the two separate standard errors."""
    rng = np.random.default_rng(2)
    rho, K = 0.9, 400
    truth = 0.3 + CVAR_N01 - 1.2 * CVAR_N01
    d, se_p, cov = [], [], 0
    for _ in range(2000):
        z1 = rng.normal(size=K)
        z2 = rho * z1 + math.sqrt(1 - rho ** 2) * rng.normal(size=K)
        a, b = 0.3 + z1, 1.2 * z2
        r = ft.cvar_difference(a, b, ALPHA)
        d.append(r["difference"])
        se_p.append(r["se"])
        cov += r["lo"] <= truth <= r["hi"]
    assert abs(np.mean(se_p) / np.std(d) - 1) < 0.06
    assert 0.92 <= cov / 2000 <= 0.97
    u = ft.cvar_difference(a, b, ALPHA, paired=False)
    sa = ft.cvar_interval(a, ALPHA)["se"]
    sb = ft.cvar_interval(b, ALPHA)["se"]
    assert abs(u["se"] - math.hypot(sa, sb)) < 1e-12
    assert np.mean(se_p) < 0.6 * u["se"]
    same = ft.cvar_difference(a, a, ALPHA)
    assert same["difference"] == 0.0 and same["se"] == 0.0
    with pytest.raises(ValueError, match="same length"):
        ft.cvar_difference(a, b[:-1], ALPHA)
