# Changelog

## 0.11.0 (2026-10-18)

The thickness errors of a whole monitored run, two fixes, faster
scoring, a third tutorial, and the tutorials in the source
distribution.

### Added

- `monitoring_errors(lam_mon_um, recipe_t_um, n_layers, sigma_signal,
  n_inc=1.0, n_sub=1.52, quantity="T", theta0_rad=0.0, pol="s",
  n_exit=None, signal_offset=0.0, max_nonlinearity=0.1)` ->
  `MonitoringErrors` (`fabtwin.monitor`): the thickness errors of a
  whole level-monitored run, with the errors of each layer carried
  into the stops of the layers after it (`monitoring_plan` treats each
  layer as if the layers below it were exact). To first order, with
  `A[j, k] = d S_j / d t_k` (the exact derivatives of
  `fabtwin.gradients`), `dt = G (e + offset)` with `G = -A^-1`, so
  `cov(dt) = sigma_signal^2 G G^T` and `mean(dt) = offset G 1`.
  Fields: the sensitivity and propagation matrices, mean, covariance,
  standard deviations and correlations (absolute and relative), the
  per-layer errors of `monitoring_plan` for comparison, and a
  nonlinearity measure; `.twin()` gives a `GaussianTwin` of the
  monitored machine (relative thickness errors, zero index errors).
  Refused: the inputs `monitoring_plan` refuses, a non-positive
  `sigma_signal`, a non-finite offset, and a run in which a layer ends
  too close to a turning point for the errors it receives
  (`nonlinearity` above `max_nonlinearity`, the layers named).
- `simulate_monitored_runs(lam_mon_um, recipe_t_um, n_layers,
  sigma_signal=None, n_runs=None, seed=0, noise=None, ...,
  signal_offset=0.0)`: the same runs simulated layer by layer without
  any linearization; each stop is found by bisection on `stack_rt` on
  the branch of the layer's monitoring curve that contains the recipe
  thickness, all runs at once. A stop the signal cannot reach on that
  branch is NaN, with a RuntimeWarning.
- README example 22 (the errors of a whole monitored run).
- `MANIFEST.in`: the source distribution now contains `examples/`
  (and `CHANGELOG.md`, `CITATION.cff`, `CONTRIBUTING.md`), so the
  tutorial tests run from an unpacked sdist instead of being skipped.

### Fixed

- Band edges on the wavelength grid. `notch_weights` and
  `bandpass_weights` dropped a grid point that lies exactly on a band
  edge (in decimal) when binary round-off put the computed edge just
  past it. Before -> after: `notch_weights` on
  `np.linspace(0.45, 0.70, 51)` with center 0.5325, half-width 0.0125,
  guard 0.02 (README example 11): 5 -> 6 stop-band points (0.545 was
  missing, because `0.5325 + 0.0125` is `0.5449999999999999`); on
  `np.linspace(0.40, 0.70, 61)` with 0.55, 0.02, 0.01: the upper pass
  band started at 0.585 -> 0.58, while the lower one already ended at
  0.52. A point within `1e-9` of the largest wavelength of an edge now
  counts as on it. Effect on README example 11: largest stop-band `T`
  at 0 degrees 0.042 -> 0.041, smooth margin `J` 0.0446 -> 0.0447. No
  other README example or tutorial changes; where no grid point lies
  within 1e-6 um of an edge the weights are unchanged exactly (200
  random grids). Tests:
  `test_v011.py::test_band_edges_on_the_grid_belong_to_their_band`,
  `test_band_edges_off_the_grid_are_unchanged`.
- `drift_test` p-values too small with few runs. A permutation that
  reproduces the observed split (or swaps two groups of equal size)
  has the observed statistic exactly, but recomputed from reordered
  rows it can come out a few units of the last digit smaller, and was
  then not counted. Before -> after (`n_perm=999`, `seed=0`, two
  groups 6 standard deviations apart, data seeds of the tests): 2 + 2
  runs, data seed 1: 0.036 -> 0.315 (with 4 runs no p-value below 1/3
  is possible); data seed 10: 0.065 -> 0.315; 3 + 3 runs, data seed 0:
  0.082 -> 0.103; 2 + 3 runs, data seed 6: 0.052 -> 0.106; under the
  null with 3 + 3 runs, the smallest of 60 seeded p-values 0.046 ->
  0.087 (the exact minimum is 0.1). A permuted statistic less than
  `1e-9` times the mean distance between the runs below the observed
  one now counts as a tie. README example 13 is unchanged. Tests:
  `test_v011.py::test_drift_test_counts_ties_in_small_groups` (5
  cases; the reference counts the permutations by their index sets,
  without floating point) and
  `test_drift_test_small_groups_never_go_below_the_exact_minimum`.

### Performance

- `evaluate_under_process`, `induced_merits` (and so
  `twin_fidelity_report`), `random_search` and the probes of
  `inverse_design`, without `model=`: all designs go through one call
  of the forward solver, side by side along its column axis (in
  chunks), instead of one call per design. Every column goes through
  the same element-wise operations as before; the tests compare with
  the 0.10.1 loops, kept as references, to 1e-12 of the largest value
  (not bit for bit: vectorized math kernels may round the last digit
  differently on other processors). Measured on one shared 2-CPU
  machine (NumPy 2.5, CPU time, median of 15 interleaved runs, two
  sessions), 41 wavelengths: `evaluate_under_process` with 500 draws
  of the reference process 2.8x (8 layers) and 2.2x (20 layers),
  sampling included; `induced_merits` with 500 error vectors 3.5x to
  4.0x (8 layers) and 2.4x to 2.7x (20 layers); 200 probes 3.1x to
  3.6x (8 layers) and 2.5x to 2.7x (20 layers).

### Tutorials

- `examples/tutorial_monitored_run.py`: carries the errors through
  whole monitored runs at every monitor wavelength from 0.40 to
  0.80 um, checks them at the best wavelength against 2000 simulated
  runs, picks the wavelength that is best for the filter's worst 10 %
  from the twin of each wavelength and confirms it with simulated runs
  (paired), and shows the effect of a 0.2 % calibration offset of the
  monitor. About 10 s. Its printed output is in the README and at the
  top of the script; `tests/test_tutorials.py` runs it and checks the
  printed numbers.

### Changed

- `fabtwin.tmm` accepts, internally only, one thickness per layer and
  wavelength column (used by the batched scoring and by
  `simulate_monitored_runs`); the public calls are unchanged.
- README: example 11's output and a note, example 22, the third
  tutorial, the new functions, refusals, checks, corrections and
  limits.

### Tests

197 tests (180 in 0.10.1): new file `test_v011.py` (16) and one more
tutorial test. Independent references: integer grid arithmetic for
the band edges; the index sets of the permutations of the drift test;
the 0.10.1 per-design loops; the closed form of one layer on a
substrate, inverted by hand; SciPy's `brentq` on `stack_rt` for
simulated runs; central finite differences of the simulated run; and
the scatter of 2000 simulated runs. The new tests add about 17 s,
about 9 s of it the new tutorial.

## 0.10.1 (2026-10-18)

A test and documentation fix. No result of the package changes.

### Fixed

- Two speed tests of 0.10.0 asked the batched computation to equal the
  per-draw loop bit for bit: the draws of `robustify`,
  `cvar_objective_and_grad` and `ru_objective_and_grad` (without
  `model=`), and the batched hand adjoint. This holds on the
  development machine. On GitHub's test machines it failed once, in
  the oldest-dependencies job (Python 3.10, NumPy 1.26.0), and the same
  code passed in two later runs there. So the last digits can differ
  between machines. The tests now allow a difference of 1e-12 of the
  largest value (`test_batched_draws_equal_the_per_draw_loop`, formerly
  `..._bit_for_bit`, and `test_batched_adjoint_rows_equal_merit_and_grad`).
- The README, the `fabtwin.adjoint` docstring and a comment in
  `fabtwin.robust` said "bit for bit" for this path. They now say
  "equal to rounding" and explain why. The 0.10.0 entry below is left
  as released; its "equal bit for bit" holds on the development
  machine only.

## 0.10.0 (2026-10-17)

Planning the optical monitoring of a run, two fixes at extreme inputs,
a faster `robustify`, and two worked tutorials.

### Added

- `monitoring_plan(lam_mon_um, recipe_t_um, n_layers, n_inc=1.0,
  n_sub=1.52, quantity="T", sigma_signal=None, theta0_rad=0.0,
  pol="s", n_exit=None, points_per_quarter_wave=32,
  max_nonlinearity=0.1)` -> `MonitoringPlan` (new module
  `fabtwin.monitor`): the level-monitoring plan of a recipe at one
  monitor wavelength, before any run (level monitoring as in G. J.
  Kopec, Optica Applicata 17(1), 33 (1987)). For every layer, in the
  order of deposition (substrate side first): the monitor signal along
  its growth (`curves`), the value at which to stop it (`cutoff`), the
  exact slope there, the number of turning points passed, and, for a
  stated signal noise, the thickness error it causes,
  `sigma_t = sigma_signal / |slope|` (first-order propagation). Where
  the second-order term is not small (`nonlinearity =
  |S''| sigma_signal / (2 S'^2)` above `max_nonlinearity`), that is,
  where a layer ends too close to a turning point, `sigma_t` is NaN
  with a RuntimeWarning instead of a number that looks precise.
  Transmission or reflection, any angle and polarization, absorbing
  layers, and a monitor that measures through the whole plate
  (`n_exit`). Refused: a non-positive wavelength or thickness,
  mismatched shapes, a quantity other than "T" or "R", a non-positive
  noise, too coarse a sampling.
- README example 21 (monitoring plans at two wavelengths).

### Fixed

- NaN for opaque layers. In a strongly absorbing layer, or a thick
  layer in which the light is evanescent, cos and sin of the complex
  phase grow like `exp(|Im delta|)`. `stack_rt`, `transmittance`,
  `reflectance` and `stack_rta_and_grads` overflowed: NaN for `R`, `T`
  and every derivative once the sum `m` of `|Im delta|` over the layers
  passed about 709, and overflow warnings from about 355. Before ->
  after: 0.1 um of index 1.46 over 10 um of `n = 1 + 6i` on glass
  (1.52), 0.5 um, normal incidence: `R` NaN -> 0.840580 (the `tmm`
  package: 0.840580), `T` NaN -> 0; 10 um of that metal alone:
  `R` NaN -> 0.9 (the Fresnel value 36/40); glass (1.5) -> 70 um of
  air -> glass at 60 degrees: `R` NaN -> 1.0. Cause: no rescaling of
  the layer matrices. Now, in every wavelength column with `m` above
  300, each layer matrix (and its derivatives) is multiplied by
  `exp(-m_i)`, which leaves `r` and every ratio of derivatives
  unchanged, and `T` is multiplied back by `exp(-2 m)`. Columns with
  `m` up to 300 take the old path, bit for bit (300 random stacks
  compared with 0.9.1: forward optics, every derivative, with and
  without `n_exit`). Tests:
  `test_v010.py::test_opaque_metal_gives_the_fresnel_closed_form`,
  `test_opaque_layer_under_a_coating_matches_tmm_package`,
  `test_thick_evanescent_gap_reflects_totally`,
  `test_opaque_rescaling_leaves_other_results_unchanged`.
- A conformal rank of 0. `_ceil_count` (0.9.0) returned 0 for
  `0 < x <= 1e-9`, so for `alpha` within about `1e-9 / (n + 1)` of 1
  the rank `ceil((n + 1)(1 - alpha))` was 0, and index `0 - 1` picked
  the largest score. Before -> after: `conformal_quantile` of the
  scores 1..10 at `alpha = 1 - 1e-12` 10.0 -> 1.0;
  `conformal_coverage_exact(10, 1 - 1e-12)` 0.0 -> 0.0909 (1/11);
  `AdaptiveConformal(|N(0,1)| x 100, seed 0, alpha=0.25,
  gamma=0.05)` after 60 runs scored 0 (`alpha_t` =
  0.9999999999999984): `half_width()` 2.3250 -> 0.0;
  `mondrian_quantiles` the same as `conformal_quantile`. A positive
  count is now at least 1. Tests:
  `test_v010.py::test_positive_count_is_never_zero`,
  `test_conformal_quantile_near_alpha_one_is_the_smallest_score`,
  `test_adaptive_conformal_half_width_just_below_alpha_one`.

### Performance

- `robustify`, `cvar_objective_and_grad` and `ru_objective_and_grad`
  without `model=`: all draws of a step go through the hand adjoint at
  once (in chunks that fit the processor cache), with the same
  arithmetic, element by element, as the per-draw loop. The results
  are equal bit for bit (`np.array_equal` against the 0.9.1 loop,
  kept in the tests, on 25 random cases; the README examples print the
  same). Measured on one 2-CPU machine (median of 5 runs, NumPy 2.5):
  8 layers, 41 wavelengths, `K = 32` (README example 3): 3.4x to 3.9x;
  10 layers, 61 wavelengths, `K = 64`: 2.4x to 2.9x; 20 layers, 161
  wavelengths, `K = 128`: 1.1x to 1.5x (the range is over repeated
  sessions on a shared machine).

### Tutorials

- `examples/tutorial_monitoring.py`: design a notch filter, scan the
  monitor wavelength with `monitoring_plan`, check the predicted
  thickness errors against 400 simulated stops per layer, and compare
  the merit CVaR at the best and the worst usable wavelength, with
  confidence intervals.
- `examples/tutorial_spectra_to_twin.py`: compare measurement plans
  with `recovery_precision`, recover the thickness errors of 24
  simulated runs from their spectra, and estimate the machine's bias
  (0.0201 +/- 0.0033 against a hidden 0.02) and scatter.
- Their printed output is in the README (new "Tutorials" section) and
  at the top of each script; `tests/test_tutorials.py` runs both and
  checks the printed numbers.

### Changed

- `stack_rta_and_grads` accepts, internally only, one thickness per
  layer and wavelength column (used by `monitoring_plan`); the public
  call is unchanged.
- README: example 21, the Tutorials section, the new function,
  refusals, checks, corrections and limits.

### Tests

180 tests (160 in 0.9.1): new files `test_v010.py` (18) and
`test_tutorials.py` (2). Independent references: exact rational
arithmetic; the Fresnel reflectance of a semi-infinite absorbing
medium and its derivatives, by hand; total internal reflection; the
`tmm` package; the unchanged `stack_BC` and the 0.9.1 per-draw loop,
kept as references; the closed form of one layer on a substrate;
partial stacks computed separately with `stack_rt`; central finite
differences; and 400 simulated monitor stops per layer, found by root
finding on `stack_rt` (scatter within 6 % of the prediction; the test
allows 12 %). The new tests add about 20 s, most of it the two
tutorials.

## 0.9.1 (2026-10-10)

Citation corrections and packaging.

### Fixed (documentation)

- `reverse.py`: Appl. Opt. 51, 245 (2012) has eight authors and is now
  cited as "A. V. Tikhonravov et al." (it was "Tikhonravov and
  Trubetskov"), in the module docstring and in two error messages. The
  author list of Appl. Opt. 51, 5543 (2012) now includes B. Romanov.
- `risk.py`: the North American Actuarial Journal 9(2), 129 (2005)
  paper is by Manistre and Hancock, in that order.
- Packaging: the license is now given as the SPDX expression
  `license = "Apache-2.0"` with `license-files = ["LICENSE"]`, and the
  old license table and license classifier are gone. Building no longer
  prints setuptools' deprecation warnings (their support ends on
  2027-02-18). Building from source now needs setuptools 77 or newer;
  installing the package is unchanged.

## 0.9.0 (2026-10-10)

Measured spectra from instrument files, measurement planning before
tool time is booked, two bug fixes, and the published paper.

### Added

- `load_spectrum_csv(path, wavelength_unit, value_unit, column=1,
  delimiter=None)` (`fabtwin.data`): reads a measured spectrum from a
  spectrophotometer's text export (header lines skipped, a trailing
  metadata block ignored, a scan from long to short wavelengths
  reversed; comma, semicolon, tab or space between columns) and
  returns wavelengths in micrometers and values as fractions. The
  units are required arguments, never guessed. Refused, with the line
  number: repeated or out-of-order wavelengths, ragged rows, numbers
  resuming after the data block was interrupted, non-finite values,
  decimal commas, a missing column, and units that contradict the
  numbers (fractions above 1.5, wavelengths in um above 100 or in nm
  below 20).
- `recovery_precision(...)` -> `RecoveryPlan` (`fabtwin.reverse`):
  the uncertainties of the thickness (and index) errors that a planned
  set of measurements would give, before measuring, from the inverse
  Fisher information `(J^T W J)^-1` of the exact Jacobian (the
  Cramer-Rao bound for independent Gaussian noise; C. R. Rao, Bull.
  Calcutta Math. Soc. 37, 81 (1945)), with correlations and the
  condition number. A plan that cannot determine the unknowns is
  refused, and the message names the invisible combination.
  `Measurement.values` may be None in a plan.
- README examples 19 (reading an instrument export and fitting it) and
  20 (comparing measurement plans).

### Fixed

- Tail counts and conformal ranks off by one through floating-point
  round-off. `ceil(alpha K)` (CVaR) and `ceil((n + 1)(1 - alpha))`
  (conformal rank) were computed with `np.ceil` on a floating-point
  product that can land just above a whole number (`0.07 * 100` is
  `7.000000000000001`), giving one value too many. Before -> after:
  `cvar(np.arange(100.0), 0.07)` 3.5 -> 3.0 (the mean of the 7
  smallest); `cvar_interval` on 400 draws at `alpha = 0.07`, `n_tail`
  29 -> 28; `cvar_objective_and_grad` with `alpha = 0.07`, `K = 100`
  averaged the 8 worst draws, now the 7 worst (0.512112 -> 0.511929 in
  the test case); `conformal_quantile` of the scores 1..149 at
  `alpha = 0.18` 124 -> 123, `conformal_coverage_exact(149, 0.18)`
  0.8267 -> 0.82, `AdaptiveConformal.half_width` 124 -> 123 on the same
  scores. Also affected through the same count: `tail_statistics`,
  `robustify` (`estimator="sort"`), `twin_jax.robustify_gan`,
  `mondrian_quantiles`. For `alpha` in steps of 0.01 and `K`, `n` up
  to 1000, 141 tail counts and 577 conformal ranks were wrong (none for
  `alpha` = 0.05, 0.1, 0.2 or 0.25); results in all other cases, and
  every README example, are unchanged. Cause: no tolerance for
  round-off; now a product within 1e-9 (relative) of a whole number is
  taken as that number (`fabtwin.risk._ceil_count`). Tests:
  `test_lab_data.py::test_tail_count_matches_exact_rational_arithmetic`,
  `test_cvar_counts_the_right_number_of_worst_values`,
  `test_cvar_objective_averages_exactly_alpha_K_draws`,
  `test_conformal_rank_is_the_exact_formula`.
- `errors_from_spectrum` and `errors_from_spectra` without a noise
  level and with zero degrees of freedom (as many measured values as
  unknowns) returned unit-weight uncertainties, as if the noise were 1
  (100 % in `T`), although the documentation says they are scaled by
  the residual, which does not exist then. Before -> after: one layer
  of 0.080 um and index 2.0 on a substrate of 1.5, one wavelength
  (0.55 um), 1 % thickness error: `sigma` 7.400 -> NaN, with a
  `RuntimeWarning`; `errors_from_spectra` the same, and with `n_boot`
  it now refuses (the bootstrap noise was 1 in `T`). The recovered
  errors are unchanged. Tests:
  `test_lab_data.py::test_zero_dof_without_noise_level_gives_nan_not_unit_noise`
  and `test_zero_dof_with_noise_level_matches_finite_difference` (with
  a noise level, dof 0 keeps working and equals a finite-difference
  closed form).

### Changed

- Input checks, for calls whose results were not meaningful: an angle
  of incidence that is not finite or is larger than pi/2 in size is
  refused by `stack_rt`, `transmittance` and `reflectance` (0.8.0
  returned the result for `pi - theta`; `theta0_rad = 2.0` gave
  R = 0.44983 for 0.1 um of index 2.0 on 1.5 at 0.5 um, the value at
  1.1416 rad); a negative `half_um` or `guard_um` in `notch_weights`,
  or `guard_um` in `bandpass_weights`, is refused (the bands
  overlapped); a non-positive `max_error` or `max_index_error` in the
  recovery functions is refused by name (it surfaced as "no multi-start
  converged"). Negative angles still give the same result as positive
  ones.
- `load_traces_csv` accepts a UTF-8 byte-order mark (as spreadsheet
  programs write) and names the line of a field that is not a number.
- Type hints on the public functions touched in this release.
- Documentation: the associated paper is published (IEEE Sensors
  Journal, early access, 8 October 2026,
  doi:10.1109/JSEN.2026.3732239). `README.md` and `CITATION.cff` cite
  it with its DOI, and the README gives a BibTeX entry. The `Paper`
  link in `pyproject.toml` points to the DOI; the companion repository
  is listed as `Paper code`. (Released here; it was listed as
  Unreleased after 0.8.0.)
- README: examples 19 and 20, the new functions, refusals, checks,
  corrections and limits; polarization `"u"` listed under units.

### Tests

160 tests (132 in 0.8.0), new file `test_lab_data.py` (28).
Independent references: exact rational arithmetic (`fractions`), the
direct mean of the smallest values, central finite differences of the
forward solver `stack_rt` for the planner's Jacobian, the scatter of
200 actual nonlinear fits to simulated noisy spectra (within 7 % of
the prediction at 0.1 % noise; the test allows 20 %), and files with
known unit conversions written by the test. The new tests add about
4 s.

## 0.8.0 (2026-09-30)

Measured coated plates (the substrate's back face), warnings when a
spectrum fit cannot be trusted, and confidence intervals for yields
and CVaRs.

### Added

- `n_exit=` in `stack_rt`, `transmittance`, `reflectance`,
  `stack_rta_and_grads`, `OpticalModel`, `errors_from_spectrum` and
  `errors_from_spectra`: a thick, non-absorbing substrate with a bare
  back face and a medium of index `n_exit` behind it. `R` and `T` are
  then those of the whole plate, as a spectrophotometer measures it;
  the multiple reflections between coating and back face are summed
  in power (incoherently), `T = Tf Tb / (1 - Rr Rb)`,
  `R = Rf + Tf Tr Rb / (1 - Rr Rb)`, per polarization. Exact
  derivatives of `R`, `T`, `A` (product and quotient rules on the
  existing adjoint, run once from each side of the coating), so
  design, robust design, scoring, correction and spectrum recovery all
  work for a plate through `model=` or `n_exit=`. Default `None` keeps
  the infinitely thick substrate of earlier versions.
- `SpectrumRecovery.at_bound`, `JointRecovery.at_bound_t` and
  `at_bound_n`: which recovered errors sit on the edge of the search
  box (from SciPy's `active_mask`); `chi2_pvalue` on both: the
  chi-square upper-tail probability of the fit when the noise level is
  given (None otherwise).
- `yield_interval` (exact Clopper-Pearson interval of a pass fraction,
  C. J. Clopper and E. S. Pearson, Biometrika 26, 404 (1934));
  `cvar_interval` (standard error and large-sample interval of a CVaR
  estimate, from the linearization of the Rockafellar-Uryasev form);
  `cvar_difference` (the same for the CVaR difference of two designs,
  paired for common random numbers or unpaired).

### Behaviour changes

No returned number changes. New `RuntimeWarning`s, for calls whose
results were already unreliable:

- `errors_from_spectrum` and `errors_from_spectra` warn when a
  recovered error sits on the search bound, and, when the noise level
  is given, when the chi-square p-value is below 1e-6. Example: the
  spectrum of README example 17 (a coated fused-silica plate with its
  back face, 0.2 % noise) fitted without `n_exit` gave, in 0.7.0 and
  now, thickness errors (-0.0735, 0.0284, 0.1211, -0.0842, 0.0312)
  against the true (0.03, -0.02, 0.015, 0.01, -0.025), chi2 = 388.3
  for 116 degrees of freedom, silently; 0.8.0 returns the same numbers
  with a warning (p = 4.5e-31). With `n_exit=1.0`: (0.0374, -0.0432,
  0.0417, -0.0018, -0.0195), chi2 p = 0.99. None of the 0.7.0 tests or
  README examples triggers either warning.
- `errors_from_spectrum` with a zero or negative `sigma_T` still
  raises the same ValueError, but no longer emits a NumPy
  divide-by-zero warning first.

With `n_exit` left out, `stack_rt` and `stack_rta_and_grads` return
the same numbers as 0.7.0 bit for bit (300 random cases compared
against the 0.7.0 code, forward optics and every gradient; the
internal Snell-invariant refactoring of `fabtwin.gradients` changed
no arithmetic). README examples 1 to 16 print exactly what they
printed with 0.7.0.

### Tests

132 tests (117 in 0.7.0). New files `test_substrate_backface.py` (8)
and `test_uncertainty.py` (7). Independent references: the
incoherent solver `inc_tmm` of the `tmm` package (to 1e-12 on 150
random plates), the closed forms of a bare and of an ideally
anti-reflection-coated plate, central finite differences, the
closed-form chi-square tail for even degrees of freedom, binomial
tails summed directly, exact binomial coverage, the closed-form CVaR
and its linearized variance for a normal distribution, and seeded
simulations (coverage of the 95 % CVaR interval asserted between
91 % and 97 % with 40 tail draws, observed 93.2 %, and between 88 %
and 96 % with 20, observed 91.1 %). The new tests add about 10 s.

### Changed

- README: examples 17 (a measured plate) and 18 (confidence intervals,
  on the designs of example 3), new glossary entries, checks,
  refusals, and Limits.

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
