"""Bring your own fabrication data: ingestion and validation.

The twins of this package never see a simulator -- they consume
(recipe, outcome) traces, which is exactly what a monitored deposition
tool accumulates (the paper's Discussion names a twin trained on real
in-situ monitoring data as the essential next step; this module is
that interface). The reference `DepositionProcess` exists to
*generate* benchmark traces, not to gate the pipeline: anything that
produces the four arrays (rt, rn, ft, fn) drops into the identical
loop.

The trace file contract (long/tidy CSV, one row per run and layer):

    run, layer, t_recipe_um, n_recipe, t_fab_um, n_fab

with `layer` running 1..N complete and consecutive within every run,
thicknesses in micrometres and indices at the platform's reference
wavelength. `load_traces_csv` enforces the contract instead of
guessing; `save_traces_csv` writes it, so the round trip is exact
(asserted in the tests). No pandas -- the reader is the standard
library's csv module on top of NumPy.

`validate_traces` separates two kinds of trouble in real data:
structural problems (shape mismatches, non-finite values,
non-positive thicknesses) RAISE, because no downstream result is
meaningful; plausibility findings (relative thickness errors or
absolute index errors beyond screening thresholds that usually mean a
unit mix-up or a mis-paired run) are RETURNED in a report for the
user to judge, because a package should not silently discard a lab's
outliers -- the heavy tail is precisely what the twin is for. The
screening thresholds are explicit keyword parameters with documented
defaults (50 percent relative thickness error, 0.5 absolute index
error), not hidden constants.
"""
from __future__ import annotations

import csv

import numpy as np

__all__ = ["load_traces_csv", "save_traces_csv", "validate_traces"]

_COLUMNS = ("run", "layer", "t_recipe_um", "n_recipe", "t_fab_um",
            "n_fab")


def save_traces_csv(path, rt, rn, ft, fn):
    """Write traces in the documented contract. Arrays are (M, N)."""
    rt, rn, ft, fn = (np.asarray(a, dtype=float)
                      for a in (rt, rn, ft, fn))
    if not (rt.shape == rn.shape == ft.shape == fn.shape) or rt.ndim != 2:
        raise ValueError("all four trace arrays must share one (M, N) "
                         "shape")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(_COLUMNS)
        for m in range(rt.shape[0]):
            for i in range(rt.shape[1]):
                w.writerow([m + 1, i + 1,
                            repr(float(rt[m, i])), repr(float(rn[m, i])),
                            repr(float(ft[m, i])), repr(float(fn[m, i]))])


def load_traces_csv(path):
    """Read traces in the documented contract; returns (rt, rn, ft, fn)
    as (M, N) arrays ordered by run then layer.

    Refusals instead of guesses: missing or extra header columns, a
    run whose layers are not exactly 1..N, or runs with differing N.
    """
    rows = {}
    with open(path, newline="") as fh:
        r = csv.reader(fh)
        try:
            header = next(r)
        except StopIteration:
            raise ValueError("empty trace file") from None
        if tuple(h.strip() for h in header) != _COLUMNS:
            raise ValueError(
                f"trace file header must be exactly {_COLUMNS}; got "
                f"{tuple(header)}")
        for line, rec in enumerate(r, start=2):
            if not rec:
                continue
            if len(rec) != 6:
                raise ValueError(f"line {line}: expected 6 fields")
            run = rec[0].strip()
            layer = int(rec[1])
            vals = tuple(float(x) for x in rec[2:])
            rows.setdefault(run, {})
            if layer in rows[run]:
                raise ValueError(f"run {run}: duplicate layer {layer}")
            rows[run][layer] = vals
    if not rows:
        raise ValueError("trace file contains no data rows")
    n_layers = None
    out = []
    for run in rows:
        layers = sorted(rows[run])
        if layers != list(range(1, len(layers) + 1)):
            raise ValueError(
                f"run {run}: layers must be exactly 1..N consecutive; "
                f"got {layers}")
        if n_layers is None:
            n_layers = len(layers)
        elif len(layers) != n_layers:
            raise ValueError(
                f"run {run}: has {len(layers)} layers where earlier "
                f"runs have {n_layers}")
        out.append([rows[run][i] for i in layers])
    arr = np.asarray(out, dtype=float)          # (M, N, 4)
    return arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]


def validate_traces(rt, rn, ft, fn, max_rel_t_err=0.5,
                    max_abs_n_err=0.5):
    """Structural checks raise; plausibility findings are returned.

    Returns a report dict: n_runs, n_layers, and `flags`, a list of
    (run_index, layer_index, kind, value) tuples for outcomes beyond
    the screening thresholds (`rel_t` for |t_fab/t_recipe - 1| >
    max_rel_t_err, `abs_n` for |n_fab - n_recipe| > max_abs_n_err) --
    values that usually indicate a unit mix-up or mis-paired run, left
    to the user's judgment rather than silently dropped.
    """
    rt, rn, ft, fn = (np.asarray(a, dtype=float)
                      for a in (rt, rn, ft, fn))
    if not (rt.shape == rn.shape == ft.shape == fn.shape) or rt.ndim != 2:
        raise ValueError("all four trace arrays must share one (M, N) "
                         "shape")
    for name, a in (("recipe thickness", rt), ("recipe index", rn),
                    ("fabricated thickness", ft),
                    ("fabricated index", fn)):
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{name} contains non-finite values")
    if np.any(rt <= 0) or np.any(ft <= 0):
        raise ValueError("thicknesses must be positive (check units)")
    flags = []
    rel_t = ft / rt - 1.0
    abs_n = fn - rn
    for m, i in zip(*np.nonzero(np.abs(rel_t) > max_rel_t_err)):
        flags.append((int(m), int(i), "rel_t", float(rel_t[m, i])))
    for m, i in zip(*np.nonzero(np.abs(abs_n) > max_abs_n_err)):
        flags.append((int(m), int(i), "abs_n", float(abs_n[m, i])))
    return dict(n_runs=int(rt.shape[0]), n_layers=int(rt.shape[1]),
                flags=flags)
