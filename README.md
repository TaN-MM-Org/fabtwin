# fabtwin

[![tests](https://github.com/TaN-MM-Org/fabtwin/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/fabtwin/actions)
[![PyPI](https://img.shields.io/pypi/v/fabtwin?cacheSeconds=3600)](https://pypi.org/project/fabtwin/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Learned generative process twins for yield-aware inverse design of
multilayer optics.** Real deposition errors are systematic,
layer-correlated, design-conditional, skewed and heavy-tailed, and
manufacturing yield is decided precisely in the tail that hand-specified
Gaussian tolerance models represent worst. `fabtwin` is the generalized
library form of the FabGAN-ID framework (Mahim, Islam, Rahman and
Mohsin, *IEEE Sensors Journal*, 2026): learn the fabrication process
from historical (recipe, outcome) traces, then robustify designs by
ascending the yield-deciding lower-tail statistic **pathwise through the
learned twin and the exact physics at once**.

> Learn what cannot be simulated; differentiate what can.

Where the paper's companion repository reproduces the paper (fixed
20-layer SiNx notch platform, JAX required end to end), `fabtwin` is the
reusable instrument: any layer count, any design box, any wavelength
grid, any cited material system, any linear-in-T merit -- and the entire
differentiable fabrication loop runs on **NumPy alone**, because the
discrete adjoint of the transfer-matrix recursion is derived by hand
rather than delegated to an autodiff framework. The learned conditional
WGAN-GP twin is an optional extra (`pip install 'fabtwin[twin]'`, JAX +
optax), and its autodiff gradients are asserted equal to the hand
adjoint to machine precision -- two independent derivations of the same
physics agreeing is the package's core cross-validation.

## What is inside

- `fabtwin.tmm` -- exact characteristic-matrix optics (Macleod) for
  arbitrary stratified stacks: normal or oblique incidence (s/p),
  absorbing layers (n + ik, gain refused), reflectance/transmittance,
  and linear-in-T merit builders (fluorescence-rejection notch and
  bandpass, per the paper's sensor front-ends). Cross-validated against
  the open `tmm` reference (Byrnes, arXiv:1603.02720) to 1e-12 and
  against closed forms: bare-interface Fresnel, the exact absentee
  half-wave layer, the exact quarter-wave antireflection transform, the
  analytic Brewster zero, and R + T = 1 at machine precision.
- `fabtwin.adjoint` -- the hand-derived exact adjoint: `merit_and_grad`
  returns dJ/d(thickness) and dJ/d(index) from two O(NL) sweeps of
  prefix/suffix products, with the dispersion `shape` either shared
  (the paper's variable-index single-material platform) or per-layer
  (N, L) -- so classic multi-material stacks use the same
  differentiable loop. Matches central finite differences at the
  paper's own accuracy figure and vanishes exactly at the closed-form
  quarter-wave optimum.
- `fabtwin.materials` -- a cited Sellmeier engine (Si3N4: Luke 2015;
  fused silica: Malitson 1965; both CC0 via refractiveindex.info) with
  validity-range refusal instead of silent extrapolation, and the
  variable-index layer construction of the single-material SiNx
  platform (Yesilyurt 2023).
- `fabtwin.process` -- the six-mechanism virtual deposition process of
  the FabGAN-ID benchmark as a parametric dataclass (systematic bias,
  index drift, intermixing, design-conditional AR(1) thickness noise,
  right-skewed index noise, rare particulates): every mechanism
  re-scalable or removable, the paper's published magnitudes as cited
  defaults, and an exact closed-form anchor when the randomness is off.
- `fabtwin.twins` -- the common error parameterization
  x = (t̃/t − 1, ñ − n) with exact round trip, and Gaussian twins
  (diagonal/full) as differentiable parametric baselines.
- `fabtwin.risk` -- CVaR (exact sorted-tail convention, Rockafellar &
  Uryasev 2000), tail statistics, the hard pass/fail filter yield, and
  Monte-Carlo scoring of a design under any process or twin.
- `fabtwin.design` -- the probe-seeded projected-Adam adjoint engine
  and its equal-budget random-search baseline; recovers the analytic
  quarter-wave optimum in the tests.
- `fabtwin.robust` -- pathwise CVaR (or mean − beta*sigma)
  robustification in pure NumPy for affinely reparameterized twins,
  with the frozen-latent gradient anchored against finite differences.
- `fabtwin.twin_jax` (the `[twin]` extra) -- the conditional,
  moment-matched WGAN-GP process twin, optional physics-in-the-loop
  tail calibration (the paper's honest finding is reproduced in the
  docstring: it helps only when traces are scarce), a JAX solver
  asserted equal to the NumPy one, and `robustify_gan`: the paper's
  Stage 2, CVaR ascent through generator and solver jointly.

## Install

```
pip install fabtwin              # NumPy core: physics, adjoint, Gaussian twins, CVaR loop
pip install 'fabtwin[twin]'      # + JAX/optax: the learned WGAN-GP twin
```

## The loop in five lines

```python
import numpy as np, fabtwin as ft

lam = np.linspace(0.40, 0.80, 161)                     # um
S = ft.dispersion_shape(lam, 0.550)                    # SiNx shape (Luke 2015)
nsub = ft.SIO2_MALITSON1965.n(lam)                     # substrate (Malitson 1965)
w, c0 = ft.notch_weights(lam, 0.532, 0.015, 0.030)     # 532-nm rejection merit
box = ft.DesignBox(20, 0.020, 0.120, 1.6, 2.4)         # fabrication box

t, n0, J, _ = ft.inverse_design(lam, S, w, c0, box, n_sub=nsub)   # nominal optimum

rng = np.random.default_rng(0)
recipes = box.sample(rng, 200)
rt, rn, ftd, fnd = ft.PAPER_PROCESS.trace_dataset(*recipes, 2, rng)  # historical traces
twin = ft.GaussianTwin(ft.errors_from_traces(rt, rn, ftd, fnd))      # or train the WGAN twin

t_rob, n_rob, hist = ft.robustify(lam, t, n0, S, w, c0, twin, box,
                                  alpha=0.05, n_sub=nsub)            # CVaR ascent
report = ft.evaluate_under_process(
    lambda a, b, K: ft.PAPER_PROCESS.ensemble(a, b, K, rng),
    t_rob, n_rob, lam, S, w, c0, K=2000, n_sub=nsub)                 # score vs the process
```

The learned twin drops in through the `[twin]` extra:
`fabtwin.twin_jax.train_wgan` on the same traces, then
`robustify_gan` ascends the identical objective through the generator.

## Status

v0.1.1 (alpha). Implemented and tested (44 tests, Python 3.10-3.13;
the JAX extra's tests skip cleanly without it): everything listed
above, with every physics claim anchored to a closed form, an
independent reference implementation, or an exact identity -- never to
a stored number. The adaptation testbench runs the complete loop on a
platform the paper never touched (a two-material Si3N4/SiO2 mirror
with per-layer dispersion, held to the textbook quarter-wave-stack
closed form) and exercises user-registered materials end to end, so
"generalizes" is a test result, not a claim. The API may change
before v1.0.

Deliberate scope, designed out with reasons rather than overlooked:

- The hand adjoint covers normal incidence and lossless (real-index)
  designs -- the regime of the FabGAN-ID loop. The oblique and
  absorbing *forward* solver is provided; its gradients are not faked.
- No specification-conditioned correction policies (the paper's Stage
  3): the paper itself scopes that study as exploratory, and a policy
  layer would import its chain-GCN encoder wholesale rather than
  generalize it. The pathwise machinery a policy trainer needs is all
  here.
- No neural forward surrogates, by *result*: the paper's protocol
  study found the exact differentiable solver strictly dominates them
  in this regime (faster, exact, with exact gradients), so shipping
  one would package a documented mistake.
- The benchmark data (300 designs / 48,300 samples / 400 traces) stays
  with the paper's companion repository and its Zenodo archive; this
  package regenerates equivalent data from `DepositionProcess` instead
  of redistributing files.

## Associated paper

T. M. Mahim, M. N. Islam, M. M. Rahman, A. S. M. Mohsin, "FabGAN-ID:
Learning the Fabrication Process for Yield-Aware Inverse Design of
Multilayer Photonic Sensor Filters", *IEEE Sensors Journal* (2026).
Companion repository:
[Learned-generative-process-twins...](https://github.com/Tanvir-Mahmud-Mahim/Learned-generative-process-twins-for-yield-aware-inverse-design-of-multilayer-photonic-sensor)
(benchmark archived on Zenodo, doi:10.5281/zenodo.21315793).

## Support and governance

The package is written and maintained by Tanvir Mahmud Mahim
(Department of Electrical and Electronic Engineering, BRAC University),
who reviews every change and takes the final decision on scope and
releases. There is no separate governance body; design questions are
discussed in the open in issues and pull requests, and the standing
rule of [CONTRIBUTING.md](CONTRIBUTING.md) binds the maintainer exactly
as it binds contributors: a change that touches physics arrives with a
test, and a constant arrives with its source.

## License and citation

Apache-2.0. Cite via [CITATION.cff](CITATION.cff) and the associated
paper above.
