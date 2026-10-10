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

`load_spectrum_csv` (new in 0.9.0) reads the other kind of lab data
the package uses: a measured transmittance or reflectance spectrum as
a spectrophotometer exports it (a text table of wavelength and %T or
%R, with header lines). The units are arguments, not guesses, and a
file the reader cannot read unambiguously is refused with the line
number, never patched up.
"""
from __future__ import annotations

import csv

import numpy as np

__all__ = ["load_traces_csv", "save_traces_csv", "validate_traces",
           "load_spectrum_csv"]

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


def load_traces_csv(path) -> tuple:
    """Read traces in the documented contract; returns (rt, rn, ft, fn)
    as (M, N) arrays ordered by run then layer.

    Refusals instead of guesses: missing or extra header columns, a
    run whose layers are not exactly 1..N, or runs with differing N.
    A field that is not a number is refused with its line number, and
    a UTF-8 byte-order mark (written by some spreadsheet programs) is
    accepted (both since 0.9.0).
    """
    rows = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
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
            try:
                layer = int(rec[1])
                vals = tuple(float(x) for x in rec[2:])
            except ValueError:
                raise ValueError(
                    f"line {line}: layer must be a whole number and the "
                    f"other fields numbers; got {rec}") from None
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


# ----------------------------------------------------------------------
# Measured spectra from an instrument export (new in 0.9.0)

_WAVELENGTH_UNITS = {"nm": 1e-3, "um": 1.0}
_VALUE_UNITS = {"percent": 1e-2, "fraction": 1.0}


def _numeric_fields(line, delimiter):
    """The fields of a line as floats, or None when the line is not a
    row of at least two numbers. A trailing delimiter (an empty last
    field, common in spreadsheet exports) is ignored."""
    s = line.strip()
    if not s:
        return None
    parts = s.split() if delimiter is None else s.split(delimiter)
    parts = [p.strip() for p in parts]
    while parts and parts[-1] == "":
        parts.pop()
    if len(parts) < 2:
        return None
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def load_spectrum_csv(path, wavelength_unit: str, value_unit: str,
                      column: int = 1,
                      delimiter: str | None = None) -> tuple:
    """Read a measured spectrum from a spectrophotometer export (new in
    0.9.0): a text file with one wavelength per row and one or more
    measured columns (for example %T and %R).

    path : the file. Leading lines that are not rows of numbers (the
        instrument's header: sample name, column titles) are skipped.
        The data are the first block of rows of numbers; the block ends
        at the first blank line or line that is not a row of numbers,
        so a metadata block written after the data (as some
        instruments do) is ignored too.
    wavelength_unit : "nm" or "um" -- REQUIRED, never guessed.
    value_unit : "percent" (0 to 100, as most instruments write %T and
        %R) or "fraction" (0 to 1) -- REQUIRED, never guessed.
    column : which column holds the values (0 is the wavelength;
        default 1, the first measured column).
    delimiter : None (default: the first of ",", ";", tab or blank
        space that splits the first data row into numbers) or the
        delimiter to use.

    Returns (lam_um, values): wavelengths in micrometers, increasing
    (a scan written from long to short wavelengths is reversed), and
    the values as fractions, exactly as measured (no clipping: noise
    can put a measured value slightly below 0 or above 1).

    Refused, with the line number where it applies, rather than
    guessed: no rows of numbers; rows with different numbers of
    columns; numbers that continue after the data block was
    interrupted (a damaged row in the middle of the data); a missing
    column; a non-finite value; a wavelength that is not positive;
    wavelengths that repeat or do not run in one direction; an unknown
    unit; decimal commas ("550,0;45,3" or "550,0<tab>45,3": convert
    the file to decimal points first; a file that uses the comma both
    between columns and as the decimal mark is ambiguous and cannot be
    detected, so check the result); and units that contradict the
    numbers: value_unit
    "fraction" with values above 1.5, wavelength_unit "um" with every
    wavelength above 100, or "nm" with every wavelength below 20.
    """
    if wavelength_unit not in _WAVELENGTH_UNITS:
        raise ValueError("wavelength_unit must be 'nm' or 'um'; got "
                         f"{wavelength_unit!r}")
    if value_unit not in _VALUE_UNITS:
        raise ValueError("value_unit must be 'percent' or 'fraction'; "
                         f"got {value_unit!r}")
    if int(column) != column or int(column) < 1:
        raise ValueError("column must be a whole number >= 1 (column 0 "
                         "is the wavelength)")
    column = int(column)
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        lines = fh.read().splitlines()
    candidates = [delimiter] if delimiter is not None else \
        [",", ";", "\t", None]
    start = delim = None
    for i, line in enumerate(lines):
        for d in candidates:
            if _numeric_fields(line, d) is not None:
                start, delim = i, d
                break
        if start is not None:
            break
    if start is None:
        hint = ""
        if any(c.isdigit() for line in lines for c in line):
            hint = ("; if the file uses decimal commas (550,0), convert it "
                    "to decimal points first")
        raise ValueError(f"{path}: no rows of numbers found{hint}")
    rows = []
    n_fields = None
    end = len(lines)
    for i in range(start, len(lines)):
        f = _numeric_fields(lines[i], delim)
        if f is None:
            end = i
            break
        if n_fields is None:
            n_fields = len(f)
        elif len(f) != n_fields:
            raise ValueError(f"line {i + 1}: {len(f)} columns where the "
                             f"data rows above have {n_fields}")
        rows.append(f)
    for i in range(end, len(lines)):
        f = _numeric_fields(lines[i], delim)
        if f is not None and len(f) == n_fields:
            raise ValueError(
                f"line {i + 1}: rows of numbers continue after the data "
                f"block ended at line {end + 1} ({lines[end].strip()!r}); "
                "a damaged row inside the data? Fix or remove it")
    if column >= n_fields:
        raise ValueError(f"column {column} does not exist: the data rows "
                         f"have {n_fields} columns (0 is the wavelength)")
    a = np.asarray(rows, dtype=float)
    lam, val = a[:, 0], a[:, column]
    bad = np.nonzero(~np.isfinite(lam) | ~np.isfinite(val))[0]
    if bad.size:
        raise ValueError(f"line {start + bad[0] + 1}: non-finite value")
    if np.any(lam <= 0):
        i = int(np.nonzero(lam <= 0)[0][0])
        raise ValueError(f"line {start + i + 1}: wavelength must be "
                         "positive")
    if lam.size < 2:
        raise ValueError("need at least two data rows")
    d = np.diff(lam)
    direction = np.sign(d[0])
    wrong = np.nonzero(np.sign(d) != direction)[0] if direction != 0 \
        else np.array([0])
    if wrong.size:
        i = int(wrong[0]) + 1
        raise ValueError(
            f"line {start + i + 1}: wavelengths must be all increasing or "
            "all decreasing, without repeats (several scans in one "
            "file?)")
    if direction < 0:
        lam, val = lam[::-1], val[::-1]
    if value_unit == "fraction" and np.max(val) > 1.5:
        raise ValueError(
            f"values up to {np.max(val):.4g} cannot be fractions of the "
            "light power; are they in percent (value_unit='percent')?")
    if wavelength_unit == "um" and np.min(lam) > 100.0:
        raise ValueError(
            f"wavelengths from {np.min(lam):.4g} um: are they in "
            "nanometers (wavelength_unit='nm')?")
    if wavelength_unit == "nm" and np.max(lam) < 20.0:
        raise ValueError(
            f"wavelengths up to {np.max(lam):.4g} nm: are they in "
            "micrometers (wavelength_unit='um')?")
    return lam * _WAVELENGTH_UNITS[wavelength_unit], \
        val * _VALUE_UNITS[value_unit]
