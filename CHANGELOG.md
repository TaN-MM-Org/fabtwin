# Changelog

## 0.7.0 (2026-09-23)

Gradients at any angle and with absorption, merits beyond w . T, an
unbiased gradient for CVaR design, drift and novelty checks, joint thickness and
index recovery, drift-robust and per-group conformal bands, in-run
correction, and one bug fix.

### Fixed

- Beyond the critical angle of a lossless substrate, `stack_rt`,
  `transmittance` and `reflectance` took the growing instead of the
  decaying square root for `n cos(theta)` (layers inside the stack are
  unaffected: either root gives the same result there). `T` is 0
  either way, and with all layers lossless `R = 1` either way; with
  absorption in the stack `R` was wrong: glass (1.52) -> 100 nm of 1.38 -> 80 nm
  of 2.1 + 0.05i -> air at 0.8 rad (s) gave R = 0.938 where the
  independent `tmm` package gives 0.888 (as 0.7.0 does). Results below
  every critical angle are bit-for-bit unchanged (286 random cases).
  The comparison with `tmm` now covers oblique, absorbing and
  beyond-critical stacks (200 random cases, 1e-12); before, it was
  normal incidence only.

### Added

- `gradients.stack_rta_and_grads`, `layer_indices`: exact derivatives
  of R, T and A with respect to thickness, index and extinction, any
  angle, s, p or unpolarized light, absorbing layers and substrates.
- `merits`: `LinearMerit` (on R, T or A), `TargetMerit`,
  `SpecMarginMerit` (J > 0 guarantees the pass/fail spec),
  `FunctionMerit`, `OpticalModel` (merit + angles + absorption),
  `model_spectra`, `model_merit_and_grad`. `inverse_design`,
  `adam_ascent`, `random_search`, `robustify`,
  `cvar_objective_and_grad`, `evaluate_under_process`,
  `induced_merits` and `twin_fidelity_report` accept `model=`.
- `robustify(estimator="ru")` and `ru_objective_and_grad`: the
  Rockafellar-Uryasev objective (its maximum over the threshold is
  the CVaR), with an unbiased minibatch gradient;
  `TwinEnsemble`: bootstrap ensemble of Gaussian twins.
- `fidelity.energy_distance`, `drift_test` (permutation test),
  `novelty_pvalues` (conformal p-values per run).
- `reverse.Measurement`, `errors_from_spectra`, `JointRecovery`:
  several spectra at once, thickness and optional index errors,
  optional parametric bootstrap; all refusals kept.
- `conformal.mondrian_quantiles` (per-group guarantee) and
  `AdaptiveConformal` (Gibbs and Candes, NeurIPS 2021: long-run miss
  rate under drift, with its deterministic bound).
- `GaussianTwin.conditional`: the exact Gaussian conditional given
  measured error components.
- `correct.reoptimize_remaining`: re-design the layers not yet
  deposited, nominal or robust (with a conditioned twin);
  `first="substrate"` (default) or `"incidence"` names the end grown
  first. `TwinEnsemble.conditional` conditions every member.
- `stack_rt`, `transmittance`, `reflectance` accept `pol="u"`
  (unpolarized: the mean of s and p).
- `design_recipes(refine=True)` (exchange improvement of the greedy
  maximin design) and `lab.maximin_distance`.

Defaults are unchanged: every valid 0.6.1 call gives the same result,
bit for bit, except the beyond-critical-angle fix above. Changed for
invalid input only: p polarization with a substrate exactly at its
critical angle is refused (it returned NaN); `errors_from_spectrum`
reports input errors (for example a wrong shape) directly instead of
"no multi-start converged".

### Tests

117 tests (78 in 0.6.1). New files: `test_gradients_merits.py`,
`test_robust_ru.py`, `test_drift_conformal_joint.py`,
`test_correct_lab.py`. Independent references: the `tmm` package, a
separate JAX implementation differentiated automatically, finite
differences, closed-form conditionals, seeded simulations. Passed on
Python 3.11 (NumPy 2.4, SciPy 1.17, JAX 0.10), with the core install
only on Python 3.12, and on the oldest allowed versions (Python 3.10,
NumPy 1.26.0, SciPy 1.11.0, JAX 0.4.30, optax 0.2.0, tmm 0.1.8).

### Changed

- README: seven new examples (10 to 16) with checked output, the new
  functions, refusals and checks, and a rewritten Limits section.

## 0.6.1 (2026-09-22)

Bug-fix and documentation release.

### Fixed

- `transmittance_and_grads` and `merit_and_grad` (and so
  `inverse_design`, `adam_ascent`, `robustify`,
  `cvar_objective_and_grad` and `errors_from_spectrum`, which call
  them) refused complex thicknesses and indices but silently took the
  real part of a complex dispersion `shape`, a complex `n_sub` array,
  or a NumPy complex scalar `n_sub` or `n_inc` (NumPy issued only a
  `ComplexWarning`). The returned `T` and
  gradients then described a different, non-absorbing stack: for
  three layers of index 2.0 and thickness 0.06 um on a substrate
  `1.5 + 0.2i`, 0.6.0 returned T = 0.807 at 0.45 um where
  `fabtwin.tmm` gives T = 0.787. A complex `shape`, `n_sub` or `n_inc`
  is now refused with a ValueError, like a complex index. Only calls
  with such complex inputs, outside the documented lossless scope,
  are affected; a Python `complex` scalar `n_sub` or `n_inc`
  previously raised a TypeError and now raises the same ValueError.
  As for thicknesses and indices already, the check is on the data
  type: a complex-typed array with zero imaginary part (for example
  `TabulatedMaterial.nk()` of a table without `k`, which 0.6.0
  accepted with a warning) is now refused too; pass the real values
  (`.n()`).

### Tests

- New `test_adjoint.py::test_absorbing_shape_or_substrate_is_refused_not_truncated`
  (fails on 0.6.0, passes now). 78 tests in total.
- The full suite was run on Python 3.10 with the oldest versions
  `pyproject.toml` allows (NumPy 1.26.0, SciPy 1.11.0, JAX 0.4.30,
  jaxlib 0.4.30, optax 0.2.0, tmm 0.1.8): all pass. CI now has an
  `oldest-dependencies` job that pins these versions.

### Changed

- README rewritten for readers outside the field: a guide to the
  terms, requirements and units, nine worked examples with their
  printed output (checked by running them), every public name listed,
  the refusals, and each test check with the tolerance the test
  actually uses.

### Corrections to earlier notes

- Version 0.6.0 (`fabtwin.conformal`: `conformal_quantile`,
  `conformal_interval`, `conformal_coverage_exact`) had no CHANGELOG
  entry; its tests are in `tests/test_conformal.py`.
- Some earlier entries and the 0.6.0 README describe checks as exact
  or "at machine precision" where the tests use a tolerance:
  "matches central finite differences at the paper's accuracy
  figure" (0.1.0; the test asserts a median relative error below
  1e-8 and a maximum below 1e-6, while the paper's figure is about
  1e-10); the adjoint gradient "vanishes exactly" at the quarter-wave
  optimum (below 1e-12); "exact zero recovery on a perfect
  deposition" (0.4.0; below 1e-8); interpolation "exact at the
  tabulated nodes" (0.3.0; to 1e-15); lossless reflectance identity
  "at machine precision" and the absorbing gap "asserted as an exact
  relation" (0.3.0; both to 1e-12); JAX autodiff equal to the hand
  adjoint "to machine precision" (0.6.0 README; to 1e-12); the
  error round trip "exact" (to 1e-15); a thickness-only twin's
  index errors "to numerical jitter" (0.3.0 README; below 1e-3). The
  0.5.0 replication check matches the predicted standard error within
  35 % (rtol 0.35) for the largest component.

## 0.5.0 (2026-09-17)

Lab adaptability: the calibration runs planned before the tool time
is booked.

- `lab.design_recipes`: deterministic greedy maximin spreading of
  calibration recipes over the `DesignBox` (Johnson, Moore and
  Ylvisaker, J. Statist. Plann. Inference 26, 131 (1990)), in
  box-normalized coordinates so a micron and an index unit are
  compared fairly and frozen parameters carry exactly zero distance.
- `lab.runs_for_twin_mean`: the run count for a target twin-mean
  accuracy, in closed form -- the standard error of each estimated
  mean-error component after M independent runs is exactly
  sqrt(variance/M) -- from a pilot trace set, refused when the pilot
  is too small to estimate the variances the plan stands on.
- Anchors: every greedy pick re-derived independently in the tests;
  designs respect the box exactly, are deterministic per seed, and
  ignore frozen axes exactly; the run-count inversion verified on
  both sides of the target; 60 seeded replications of the shipped
  PAPER_PROCESS match the predicted standard error.

## 0.4.0 (2026-09-13)

Reverse-engineering release: per-layer thickness errors from one
measured spectrum, with the reliability analysis built in.

### Added

- `errors_from_spectrum` / `SpectrumRecovery`: recover the per-layer
  relative thickness errors of a fabricated stack from a single
  measured transmittance spectrum and the recipe -- weighted least
  squares through the exact transfer-matrix physics with the exact
  hand-derived adjoint Jacobian, multi-start against local minima,
  per-layer uncertainties from the Jacobian (residual-scaled when no
  noise level is given), chi-square and conditioning reported.
  Reliability is refused, not guessed, three ways (Tikhonravov and
  Trubetskov, Appl. Opt. 51, 245 (2012); Amotchkina, Trubetskov,
  Pervak and Tikhonravov, Appl. Opt. 51, 5543 (2012)): fewer
  spectral points than layers; a singular Jacobian, pinned in the
  tests by the exact degeneracy that adjacent same-index layers
  enter the transfer matrix only through their thickness sum; and
  distinct error vectors fitting the spectrum equally well.
  Indices stay fixed at the recipe values, deliberately: joint
  thickness-and-index recovery from one normal-incidence spectrum is
  the classically unreliable problem the cited papers dissect.
- Anchors: noise-free round trip to 1e-6 through the exact physics;
  exact zero recovery on a perfect deposition; uncertainty
  calibration on seeded noise (z-scores bounded, chi2 consistent
  with dof); all three refusals exercised.

## 0.3.0 (2026-09-12)

Experimental-adaptability release: the three inputs a laboratory
actually has -- a measured dispersion table, a fixed set of deposited
materials, and a reflectance specification -- now enter the loop
directly.

### Added

- `TabulatedMaterial`: measured n(lam) tables (ellipsometry output,
  vendor datasheets, refractiveindex.info tabulations) used directly,
  with no Sellmeier fit required. Shape-preserving PCHIP
  interpolation, refusal outside the tabulated range (the same rule
  as the Sellmeier built-ins), and a mandatory `reference`. An
  optional extinction column feeds the absorbing forward solver
  through `.nk()`; `.n()` refuses an absorbing table rather than
  silently dropping k, because the design path is lossless. Anchors:
  exact at the tabulated nodes; between nodes, a table sampled from
  the cited Malitson Sellmeier agrees with the independent closed
  form; the whole design path (shape, solver, hand adjoint vs finite
  differences) runs on a tabulated material end to end.
- Per-layer and frozen design boxes: every `DesignBox` bound now
  accepts a scalar or an (N,) array, and a bound pair with lo == hi
  freezes that parameter -- so a real multi-material stack, whose
  indices are deposited materials rather than design variables,
  enters the identical engine. `DesignBox.thickness_only` builds the
  common case. Anchors: the design engine and the CVaR robustifier
  return a frozen index profile EXACTLY; per-layer sampling and
  clipping respect each layer's window; a fully frozen box is
  refused.
- `weights_from_reflectance`: reflectance-based linear merits
  (mirrors, high reflectors) converted to transmittance weights by
  the exact lossless identity R = 1 - T, so they run through the
  same hand adjoint unchanged. Anchors: lossless agreement at
  machine precision through the solver's R + T = 1; on an absorbing
  stack the identity fails by exactly w_R . A (asserted as an exact
  relation), which is the documented scope, and the quarter-wave
  (HL)^3 mirror scores its closed-form admittance reflectance
  through this path.
- The thickness-only metrology path (a tool that logs thickness but
  not per-run index: set n_fab = n_recipe in the trace contract) is
  now documented and anchored: the fitted twin invents no index
  errors beyond numerical jitter, and robustification on a
  frozen-index box consumes exactly the information the lab has.

### Changed

- `FabTwinConfig` validates its bounds (strictly increasing
  scalars): the learned twin's recipe conditioner normalizes by
  hi - lo, so frozen or per-layer bounds -- a design-space feature of
  `DesignBox` -- are refused there with an explanation instead of
  producing division by zero.

## 0.2.0 (2026-09-10)

Real-data release: the pipeline the paper names as the essential next
step -- a twin trained and scored on a lab's own traces -- is now a
documented, validated pathway.

### Added

- `fabtwin.data`: `load_traces_csv` / `save_traces_csv` on a
  documented tidy trace-file contract (run, layer, t_recipe_um,
  n_recipe, t_fab_um, n_fab) with an exact round trip; malformed
  files (wrong header, layer gaps, duplicates, ragged runs) are
  refused, not guessed at. `validate_traces` raises on structural
  problems (non-finite, non-positive, shape mismatch) and reports
  plausibility flags (screening defaults documented as such) instead
  of silently dropping outliers.
- `fabtwin.fidelity`: held-out twin scoring for data with no oracle.
  `moment_errors` (pooled mean/covariance/correlation discrepancies),
  `distribution_distances` (|d mean|, |d P|, |d CVaR| in the risk
  module's exact conventions, plus 1-D Wasserstein), `induced_merits`
  (error vectors -> merit samples through the exact solver) and
  `twin_fidelity_report` (the paper's Table I(A) statistic types,
  averaged over a calibration design bank). Anchors: every distance
  exactly zero against self; W1 equal to SciPy's independent
  implementation; tail statistics identical to `fabtwin.risk`; a
  deliberately mean-shifted twin scores strictly worse than a fitted
  one on held-out data.
- End-to-end bring-your-own-data test: CSV in, validate, split, fit,
  held-out fidelity report, CVaR robustification.
- README: "Bring your own fabrication data" section with the data
  contract and the honest scope statement (held-out fidelity
  certifies recorded behavior only; the paper's yield gains are
  within its virtual setting; production twins need periodic
  retraining as the tool drifts).


## 0.1.1 (2026-09-10)

Adaptation release: the differentiable loop now covers classic
multi-material stacks, and generalization is a test result.

### Added

- Per-layer dispersion: the `shape` argument of the adjoint, the JAX
  solver, the design engine and the scoring helpers now accepts a
  per-layer (N, L) array as well as the shared (L,) shape, so
  two-material (and any-material) stacks run through the identical
  differentiable loop; mismatched shapes are refused.
- Adaptation testbench (7 new anchors): the (HL)^p quarter-wave
  mirror against the independent textbook closed form
  R = ((1 - Y)/(1 + Y))^2 with Y = (nH/nL)^(2p) n_sub, to 1e-12 for
  p = 1, 3, 6; per-layer adjoint vs central finite differences on a
  dispersive Si3N4/SiO2 stack; bitwise reduction of per-layer to
  shared shapes; JAX/NumPy per-layer solver agreement to 1e-12;
  user-registered `SellmeierMaterial` end to end (range refusal,
  shape identity, adjoint-vs-FD); and the complete
  design -> traces -> twin -> CVaR robustification -> scoring loop on
  the adapted platform.


## 0.1.0 (2026-09-10)

Initial release: the generalized library form of the FabGAN-ID
framework (Mahim et al., IEEE Sensors Journal, 2026).

- Exact characteristic-matrix optics for arbitrary stacks: normal and
  oblique incidence (s/p), absorbing layers (n + ik; gain refused),
  linear-in-T merit builders. Anchors: bare-interface Fresnel, exact
  absentee half-wave and quarter-wave transforms, R + T = 1 to 1e-12,
  the analytic Brewster zero, s = p at normal incidence bitwise, and
  cross-validation against the open `tmm` reference to 1e-12.
- Hand-derived discrete adjoint (no autodiff framework): exact
  dJ/d(thickness), dJ/d(index) via prefix/suffix sweeps; matches
  central finite differences at the paper's accuracy figure, vanishes
  exactly at the closed-form quarter-wave optimum, and equals JAX
  reverse-mode autodiff to 1e-12 when the twin extra is installed.
- Cited Sellmeier materials (Luke 2015 Si3N4, Malitson 1965 fused
  silica; CC0 refractiveindex.info) with validity-range refusal, and
  the variable-index layer construction (Yesilyurt 2023).
- Parametric six-mechanism virtual deposition process with the
  paper's published magnitudes as cited defaults and an exact
  closed-form deterministic anchor; historical trace generation.
- Gaussian process twins (diagonal/full) on the common error
  parameterization (exact round trip), with exact reparameterization.
- Risk statistics: sorted-tail CVaR (exact on hand-computable sets),
  tail statistics, hard pass/fail filter yield, Monte-Carlo design
  scoring under any process or twin.
- Probe-seeded projected-Adam adjoint inverse design (recovers the
  analytic quarter-wave optimum; beats equal-budget random search)
  and pathwise CVaR / mean-variance robustification in pure NumPy,
  with frozen-latent gradients anchored against finite differences.
- Optional [twin] extra (JAX + optax): conditional moment-matched
  WGAN-GP process twin, optional physics-in-the-loop tail
  calibration, a JAX solver asserted equal to the NumPy one, and
  pathwise CVaR ascent through generator and solver jointly.
- Deliberate scope (reasons in README): no adjoint for
  oblique/absorbing stacks, no correction-policy layer, no neural
  surrogates (excluded by the paper's own protocol-study result), no
  benchmark-file redistribution.
