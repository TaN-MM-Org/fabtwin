"""Calibration-planning anchors: the greedy maximin design is
re-derived pick by pick in the test (the greedy rule is an exact,
recomputable invariant); designed recipes respect the box exactly and
are deterministic per seed; frozen parameters carry exactly zero
distance; the run-count inversion is an exact closed form verified on
both sides of the target; and seeded Monte-Carlo replications of the
shipped PAPER_PROCESS match the predicted standard error of the
estimated mean."""
import numpy as np
import pytest

from fabtwin import (DesignBox, PAPER_PROCESS, design_recipes,
                     errors_from_traces, runs_for_twin_mean)

BOX = DesignBox(6, 0.04, 0.16, 1.7, 2.1)


def _normalized(t, n, box):
    lo = np.concatenate([np.broadcast_to(box.t_lo, (box.n_layers,)),
                         np.broadcast_to(box.n_lo, (box.n_layers,))])
    hi = np.concatenate([np.broadcast_to(box.t_hi, (box.n_layers,)),
                         np.broadcast_to(box.n_hi, (box.n_layers,))])
    x = np.hstack([t, n])
    span = hi - lo
    live = span > 0
    out = np.zeros_like(x)
    out[:, live] = (x[:, live] - lo[live]) / span[live]
    return out


def test_design_respects_box_and_is_deterministic():
    t, n = design_recipes(BOX, 8, seed=4)
    assert t.shape == (8, 6) and n.shape == (8, 6)
    tc, nc = BOX.clip(t, n)
    assert np.array_equal(t, tc) and np.array_equal(n, nc)
    t2, n2 = design_recipes(BOX, 8, seed=4)
    assert np.array_equal(t, t2) and np.array_equal(n, n2)


def test_greedy_maximin_invariant_recomputed():
    """Re-derive every pick independently: the first recipe is the
    pool point nearest the centroid, and each later one maximizes the
    minimum distance to the chosen set."""
    rng = np.random.default_rng(9)
    tp, np_ = BOX.sample(rng, 64)
    pool = _normalized(tp, np_, BOX)
    t, n = design_recipes(BOX, 5, n_candidates=64, seed=9)
    x = _normalized(t, n, BOX)
    centroid = pool.mean(axis=0)
    first = int(np.argmin(np.sum((pool - centroid) ** 2, axis=1)))
    assert np.allclose(x[0], pool[first], atol=1e-12)
    chosen = [first]
    for step in range(1, 5):
        d = np.array([min(np.linalg.norm(p - pool[c]) for c in chosen)
                      for p in pool])
        d[chosen] = -np.inf
        want = int(np.argmax(d))
        assert np.allclose(x[step], pool[want], atol=1e-12)
        chosen.append(want)


def test_frozen_parameters_carry_no_distance():
    """A thickness-only box: the frozen index axes must not influence
    the design, so two boxes differing only in the frozen index value
    pick identical thicknesses."""
    b1 = DesignBox.thickness_only(4, 0.05, 0.15, 1.9)
    b2 = DesignBox.thickness_only(4, 0.05, 0.15, 2.4)
    t1, n1 = design_recipes(b1, 6, seed=1)
    t2, n2 = design_recipes(b2, 6, seed=1)
    assert np.array_equal(t1, t2)
    assert np.allclose(n1, 1.9) and np.allclose(n2, 2.4)


def test_runs_inversion_exact_two_sided():
    rng = np.random.default_rng(5)
    t0 = np.full(6, 0.08)
    n0 = np.full(6, 1.9)
    rt, rn, ft, fn = PAPER_PROCESS.trace_dataset([t0], [n0], 64, rng)
    x = errors_from_traces(rt, rn, ft, fn)
    target = 0.004
    m, se = runs_for_twin_mean(target, x)
    var = np.var(x, axis=0, ddof=1)
    assert m == int(np.ceil(var.max() / target ** 2))
    assert np.allclose(se, np.sqrt(var / m), rtol=0, atol=1e-15)
    assert float(se.max()) <= target
    if m > 1:
        assert float(np.sqrt(var.max() / (m - 1))) > target


def test_predicted_standard_error_matches_replications():
    """60 seeded replications of an M-run calibration on the shipped
    PAPER_PROCESS: the scatter of the estimated mean error must match
    the predicted sqrt(var/M) within statistical tolerance."""
    t0 = np.full(4, 0.08)
    n0 = np.full(4, 1.9)
    rng = np.random.default_rng(11)
    rt, rn, ft, fn = PAPER_PROCESS.trace_dataset([t0], [n0], 400, rng)
    pilot = errors_from_traces(rt, rn, ft, fn)
    m = 40
    pred = np.sqrt(np.var(pilot, axis=0, ddof=1) / m)
    means = []
    for _ in range(60):
        rt, rn, ft, fn = PAPER_PROCESS.trace_dataset([t0], [n0], m,
                                                     rng)
        means.append(errors_from_traces(rt, rn, ft, fn).mean(axis=0))
    emp = np.std(np.array(means), axis=0, ddof=1)
    # compare the worst component (the one the plan is sized by)
    j = int(np.argmax(pred))
    assert np.isclose(emp[j], pred[j], rtol=0.35)


def test_input_refusals():
    with pytest.raises(ValueError, match="n_recipes"):
        design_recipes(BOX, 0)
    with pytest.raises(ValueError, match="n_candidates"):
        design_recipes(BOX, 10, n_candidates=5)
    with pytest.raises(ValueError, match="positive"):
        runs_for_twin_mean(0.0, np.zeros((10, 4)))
    with pytest.raises(ValueError, match="pilot"):
        runs_for_twin_mean(0.01, np.zeros((3, 4)))
    with pytest.raises(ValueError, match="zero variance"):
        runs_for_twin_mean(0.01, np.zeros((10, 4)))
