# Changelog

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
