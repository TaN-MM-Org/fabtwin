"""Conformal anchors: the quantile is the exact rank formula
recomputed independently; seeded simulation with continuous scores
matches the EXACT closed-form coverage ceil((n+1)(1-alpha))/(n+1)
within binomial tolerance, inside the published two-sided guarantee;
an end-to-end run on the shipped PAPER_PROCESS certifies twin-free
merit intervals at their stated level; and uncertifiable levels are
refused with the minimum named."""
import numpy as np
import pytest

from fabtwin import (PAPER_PROCESS, conformal_coverage_exact,
                     conformal_interval, conformal_quantile,
                     errors_from_traces)


def test_quantile_is_exact_rank_formula():
    rng = np.random.default_rng(2)
    s = rng.exponential(1.0, 50)
    for alpha in (0.05, 0.1, 0.25):
        q = conformal_quantile(s, alpha)
        k = int(np.ceil((s.size + 1) * (1.0 - alpha)))
        assert q == float(np.sort(s)[k - 1])


def test_exact_coverage_by_seeded_simulation():
    n, alpha = 39, 0.1
    p_exact = conformal_coverage_exact(n, alpha)
    assert 1 - alpha <= p_exact <= 1 - alpha + 1.0 / (n + 1)
    rng = np.random.default_rng(17)
    hits, trials = 0, 4000
    for _ in range(trials):
        s = np.abs(rng.standard_normal(n))
        q = conformal_quantile(s, alpha)
        hits += abs(float(rng.standard_normal())) <= q
    p_hat = hits / trials
    se = np.sqrt(p_exact * (1 - p_exact) / trials)
    assert abs(p_hat - p_exact) < 4.0 * se


def test_end_to_end_on_paper_process():
    """Hold out real (simulated-process) runs, certify the mean-error
    prediction of thickness errors: the fraction of fresh runs inside
    the interval must reach the exact expected coverage within
    binomial tolerance -- no trust in any twin required."""
    t0 = np.full(4, 0.08)
    n0 = np.full(4, 1.9)
    rng = np.random.default_rng(23)
    rt, rn, ft, fn = PAPER_PROCESS.trace_dataset([t0], [n0], 60, rng)
    x_cal = errors_from_traces(rt, rn, ft, fn)
    pred = x_cal.mean(axis=0)                  # the "twin" prediction
    scores = np.max(np.abs(x_cal - pred), axis=1)
    alpha = 0.1
    q = conformal_quantile(scores, alpha)
    lo, hi = conformal_interval(pred, q)
    hits, trials = 0, 400
    for _ in range(trials):
        rt, rn, ft, fn = PAPER_PROCESS.trace_dataset([t0], [n0], 1,
                                                     rng)
        x_new = errors_from_traces(rt, rn, ft, fn)[0]
        hits += np.all((x_new >= lo) & (x_new <= hi))
    p_hat = hits / trials
    p_exact = conformal_coverage_exact(scores.size, alpha)
    se = np.sqrt(p_exact * (1 - p_exact) / trials)
    assert p_hat > 1 - alpha - 4.0 * se


def test_refusals():
    with pytest.raises(ValueError, match="at least"):
        conformal_quantile([0.1, 0.2, 0.3], alpha=0.05)
    with pytest.raises(ValueError, match="alpha"):
        conformal_quantile([0.1] * 30, alpha=0.0)
    with pytest.raises(ValueError, match=">= 0"):
        conformal_quantile([-0.1] * 30)
    with pytest.raises(ValueError, match=">= 0"):
        conformal_interval([1.0], -1.0)
    with pytest.raises(ValueError, match="certifiable"):
        conformal_coverage_exact(3, 0.05)
