# Contributing to fabtwin

Thank you for considering a contribution. This project follows one rule
above all others, the same rule its releases follow:

**A change that touches physics arrives with a test, and a constant
arrives with its source.**

## Reporting problems

Open an issue at https://github.com/TaN-MM-Org/fabtwin/issues. For a
suspected wrong number, please include the exact inputs, the value you
got, the value you expected, and the reason you expected it (a paper, a
closed form, an independent code). Wrong-number reports are the most
valuable issues a research code can receive and are treated with
priority.

## Pull requests

1. Fork, create a feature branch, and keep the change focused.
2. Run the test suite first: `pip install -e .[test]` then `pytest`.
   All tests must pass before and after your change.
3. New physics or new behavior needs a new test that fails without your
   change. Where a closed-form limit exists, test against it rather than
   against a stored number.
4. New material constants or coefficients must carry a full citation in
   the mandatory `reference` field and in the docstring; uncited
   numbers are not merged, whatever their provenance.
5. Keep dependencies minimal (NumPy + SciPy core; JAX only inside the optional [twin] extra); adding one requires
   discussion in an issue first.

## Style

Plain, documented Python. Docstrings state conventions (units, sign
conventions, energy references) explicitly, because silent convention
mismatches are the dominant failure mode of scientific code.

## Licensing and credit

Contributions are accepted under Apache-2.0, the project license.
Contributors are acknowledged in release notes; substantial contributors
are added to CITATION.cff.

## Conduct

Be professional and factual; critique code and numbers, not people.
