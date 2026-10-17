"""The tutorials in examples/ run, print what their docstrings (and the
README) say, and their results pass checks that do not depend on the
printed numbers: simulated scatter against prediction within its
standard error, a recovered bias against the hidden machine's."""
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
NUM = re.compile(r"[-+]?\d+\.\d+|[-+]?\d+")


def _run(name):
    path = EXAMPLES / name
    if not path.exists():                  # e.g. tests run from an sdist
        pytest.skip(f"{path} is not here (examples/ is in the repository)")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    out = subprocess.run([sys.executable, str(path)], capture_output=True,
                         text=True, env=env, check=True, timeout=300).stdout
    doc = ast.get_docstring(ast.parse(path.read_text()))
    block = doc[doc.index(":\n\n", doc.index("Printed output")) + 3:]
    want = [line[4:] if line.startswith("    ") else line
            for line in block.rstrip("\n").split("\n")]
    return out.rstrip("\n").split("\n"), want


def _same_numbers(got, want):
    """Every printed number equals the documented one to within one unit
    of its last printed digit (the text around them must be equal)."""
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert NUM.sub("#", g) == NUM.sub("#", w), (g, w)
        for a, b in zip(NUM.findall(g), NUM.findall(w)):
            dec = len(b.split(".")[1]) if "." in b else 0
            assert abs(float(a) - float(b)) <= 1.0001 * 10.0 ** -dec, (g, w)


def test_tutorial_monitoring():
    got, want = _run("tutorial_monitoring.py")
    _same_numbers(got, want)
    text = "\n".join(got)
    assert "best monitor wavelength 0.470 um" in text
    rows = [list(map(float, NUM.findall(line))) for line in got
            if re.match(r"\s+\d+\s+\d+\s+\d", line)]
    assert len(rows) == 8
    for _, _, pred, sim, se in rows:
        assert abs(sim - pred) <= 4 * se + 1e-3
    lo = float(re.search(r"interval (\S+) to", text).group(1))
    assert lo > 0                       # the best wavelength is better


def test_tutorial_spectra_to_twin():
    got, want = _run("tutorial_spectra_to_twin.py")
    _same_numbers(got, want)
    text = "\n".join(got)
    rms = float(re.search(r"rms (\S+) over", text).group(1))
    assert 0.8 < rms < 1.2
    m = re.search(r"pooled thickness bias (\S+) \+/- (\S+)", text)
    bias, se = float(m.group(1)), float(m.group(2))
    assert abs(bias - 0.02) < 3 * se
    twin = np.array(NUM.findall(re.search(r"twin scatter per layer:(.*)",
                                          text).group(1)), float)
    true = np.array(NUM.findall(re.search(r"same 24 runs:(.*)",
                                          text).group(1)), float)
    np.testing.assert_allclose(twin, true, rtol=0.1)


if __name__ == "__main__":                       # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
