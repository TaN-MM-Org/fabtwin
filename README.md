# fabtwin

[![tests](https://github.com/TaN-MM-Org/fabtwin/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/fabtwin/actions)
[![PyPI](https://img.shields.io/pypi/v/fabtwin?cacheSeconds=3600)](https://pypi.org/project/fabtwin/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22697049-blue)](https://doi.org/10.5281/zenodo.22697049)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

`fabtwin` is a Python package for designing **multilayer optical
filters that still work after they are made**. A filter of this kind
is a stack of thin transparent layers (tens of nanometres each) on a
glass substrate; the thickness and refractive index of each layer
decide which colours of light pass and which are blocked. A
deposition machine never builds exactly the stack you asked for. The
package learns how *your* machine tends to err from its past runs,
and then changes the design so that the worst few percent of the
devices it makes are as good as possible, because those decide how
many devices pass the specification.

It answers questions such as:

- What does this layer stack transmit and reflect, at each wavelength?
- Which layer thicknesses and indices give the best filter, if
  fabrication were perfect?
- Given logs of past runs (the recipe sent to the machine and what came
  out), what does a statistical model of the machine's errors look
  like, and how well does it match runs it has not seen?
- Which design keeps its worst 5 or 10 % of fabricated devices best?
- From one measured transmission spectrum of a finished stack, how far
  was each layer from its target thickness, and can that even be told
  apart?
- Which recipes should the calibration runs use, and how many runs are
  needed?
- What error band around a prediction holds for a stated fraction of
  new runs, without trusting any model?

The optics are computed with the standard exact transfer-matrix
method, and the design gradients with a derivative derived by hand
(the "adjoint"), so the core runs on NumPy and SciPy alone. It is the
generalized library form of the FabGAN-ID framework (Mahim, Islam,
Rahman and Mohsin, *IEEE Sensors Journal*, 2026). An optional extra
adds the paper's learned generative error model (a conditional
WGAN-GP) in JAX. The guiding rule, from the paper:

> Learn what cannot be simulated; differentiate what can.

The main results are checked by automated tests against closed-form
answers, an independent optics code, or a second independent
calculation (see [How the results are checked](#how-the-results-are-checked)).
When an input is outside what a function can answer reliably, it
raises an error that says why, instead of returning a number that
looks fine but is not.

## Contents

- [A short guide to the words used here](#a-short-guide-to-the-words-used-here)
- [Install and requirements](#install-and-requirements)
- [Examples](#examples) (each with the output it prints)
- [What is in the package](#what-is-in-the-package)
- [When it refuses, and why](#when-it-refuses-and-why)
- [How the results are checked](#how-the-results-are-checked)
- [Corrections in earlier versions](#corrections-in-earlier-versions)
- [Limits](#limits)
- [Where it comes from](#where-it-comes-from)
- [Citing, support and license](#citing-support-and-license)

## A short guide to the words used here

- **Stack, layer, substrate** -- the filter is `N` layers on a
  substrate. Layer `i` has a thickness `t_i` (micrometres) and a
  refractive index `n_i`. Light arrives from the incidence medium
  (air, index 1, by default).
- **Transmittance `T` and reflectance `R`** -- the fraction of light
  power that passes through, or bounces back, at each wavelength.
  For layers that absorb nothing, `R + T = 1`.
- **Normal and oblique incidence; s and p polarization** -- light
  arriving straight on (normal) or at an angle (oblique). At an angle,
  the result depends on the direction in which the light's electric
  field oscillates: across the plane of incidence (`"s"`) or within it
  (`"p"`).
- **Transfer-matrix method (TMM)** -- the standard exact calculation
  of `R` and `T` for a layer stack: one 2x2 matrix per layer,
  multiplied together (Macleod, *Thin-Film Optical Filters*).
- **Dispersion** -- the index changes with wavelength. A **Sellmeier
  formula** is a fitted equation for `n(wavelength)`, valid only over
  the range it was fitted on. The package describes a layer's index
  as `n_i(lam) = n0_i * S(lam)`, where `n0_i` is the index at a
  reference wavelength and `S` is the **dispersion shape** of the
  material (exactly 1 at the reference wavelength).
- **Merit `J`** -- one number scoring a spectrum, higher is better. It
  is always a weighted sum `J = w . T + const`. The built-in **notch**
  merit (block one laser line, pass everything else) is
  `J = 0.5 * (mean T in the pass bands) + 0.5 * (mean of 1 - T in the stop band)`:
  1 for a perfect notch, 0 for the exact opposite.
- **Gradient and adjoint** -- the gradient says how `J` changes when
  each thickness or index changes a little. The **adjoint** is a way
  to get all of these at once, from two passes through the stack (one
  from the front, one from the back), however many layers there are.
  Here it is written out by hand, not produced by an
  automatic-differentiation library (software that derives gradients
  from code automatically).
- **Closed form, finite difference, standard error** -- a closed-form
  answer is one given by an exact textbook formula. A (central) finite
  difference estimates a derivative by changing one input slightly up
  and down and comparing the two results. The standard error of an
  average is the typical size of its random error.
- **Design box** -- the allowed range of each thickness and index. A
  range with equal lower and upper limits **freezes** that parameter
  (for example, the index of a fixed deposited material).
- **Recipe, run, trace** -- a recipe is the stack you ask the machine
  for; a run is one deposition; a trace is the pair (recipe, what was
  actually made) for one run.
- **Error vector `x`** -- one run's errors: the relative thickness
  errors `t_made / t_recipe - 1` of every layer, followed by the index
  errors `n_made - n_recipe`. Length `2N`.
- **Process twin** -- a statistical model that produces realistic
  error vectors, fitted to traces. `GaussianTwin` is a multivariate
  normal model; the optional learned twin is a **conditional
  WGAN-GP** (Wasserstein generative adversarial network with gradient
  penalty), a neural network trained to produce error vectors that
  look like the real ones, and whose errors may depend on the recipe.
- **Reference process** -- `PAPER_PROCESS`, a simulated deposition
  machine with six kinds of error, used here to make example traces.
  It stands in for a real machine; it is not one.
- **CVaR (conditional value at risk)** -- for a set of `K` merit
  values from simulated fabricated devices, `CVaR_alpha` is the mean
  of the `ceil(alpha * K)` smallest. With `alpha = 0.1` it is "the
  average of the worst 10 %". **Robustifying** a design means changing
  it to raise this number.
- **Yield** -- the fraction of devices that meet a pass/fail
  specification.
- **Conformal prediction** -- a way to turn held-out errors into an
  error band that contains a stated fraction of new runs, without
  assuming any model is right. It needs the new runs to be
  **exchangeable** with the held-out ones (roughly: from the same,
  unchanged process).

## Install and requirements

```
pip install fabtwin              # core: optics, adjoint, Gaussian twins, CVaR design
pip install 'fabtwin[twin]'      # adds JAX and optax for the learned WGAN-GP twin
```

The core needs Python 3.10 or newer, NumPy 1.26 or newer and SciPy
1.11 or newer. The `[twin]` extra adds JAX 0.4.30 or newer and optax
0.2 or newer; nothing outside `fabtwin.twin_jax` imports JAX.

Units and conventions:

- Wavelengths and thicknesses are in **micrometres** (0.532 means
  532 nm).
- Indices of design layers are given at a reference wavelength and
  are real numbers. An absorbing material is written `n + i k` with
  `k >= 0`; the forward optics accept it, the design and gradient path
  does not (see [Limits](#limits)).
- Angles are in radians; polarization is `"s"` or `"p"`.
- Error vectors are `(relative thickness errors, absolute index
  errors)`, as defined above.
- Merits are higher-is-better, and CVaR is taken over the **lower**
  tail (the worst devices).

## Examples

Each example below runs as written, and the output shown is what it
printed with fabtwin 0.6.1 (NumPy 2.4, SciPy 1.17, JAX 0.10 on
Linux). Wavelength ranges, stacks, error sizes and noise levels are
illustrative values chosen for the examples, not recommended designs.
The built-in materials and `PAPER_PROCESS` carry their own
literature references. All examples except example 8 need only the
core install; example 8 needs the `[twin]` extra. Each ran in under
15 seconds.

### 1. Transmittance of a stack, checked against textbook answers

```python
import numpy as np
import fabtwin as ft

lam0 = 0.550                          # design wavelength, um
lam = np.array([lam0])

# One layer on a substrate of index 4.0. The ideal anti-reflection
# coating has index sqrt(1.0 * 4.0) = 2.0 and is a quarter wave thick.
n_f = 2.0
t = np.array([lam0 / (4 * n_f)])      # 0.06875 um
R, T = ft.stack_rt(lam, t, np.array([[n_f]]), n_inc=1.0,
                   n_sub=np.array([4.0]))
print(f"quarter-wave coating: R = {R[0]:.12f}, T = {T[0]:.12f}")

# Six alternating quarter-wave layers (HL)^3 make a mirror
# (H = high index, L = low index).
nH, nL, ns = 2.1, 1.5, 1.46
n = np.array([nH, nL] * 3)
R, T = ft.stack_rt(lam, lam0 / (4 * n), n[:, None], 1.0,
                   np.array([ns]))
Y = (nH / nL) ** 6 * ns               # textbook closed form
print(f"(HL)^3 mirror: R = {R[0]:.6f}, closed form {((1 - Y) / (1 + Y)) ** 2:.6f}")
```

```
quarter-wave coating: R = 0.000000000000, T = 1.000000000000
(HL)^3 mirror: R = 0.694285, closed form 0.694285
```

`n_layers` is `(N,)` for indices that do not change with wavelength,
or `(N, L)` for one value per layer and wavelength. `transmittance`
and `reflectance` return `T` or `R` alone; oblique incidence is set
with `theta0_rad` and `pol`.

### 2. Exact design gradients from the hand-written adjoint

```python
import numpy as np
import fabtwin as ft

lam = np.linspace(0.40, 0.80, 161)                # wavelengths, um
S = ft.dispersion_shape(lam, 0.550)               # Si3N4 shape, 1 at 0.55 um
nsub = ft.SIO2_MALITSON1965.n(lam)                # fused-silica substrate
w, c0 = ft.notch_weights(lam, 0.532, 0.015, 0.030)

rng = np.random.default_rng(0)
t = rng.uniform(0.020, 0.120, 20)                 # 20 thicknesses, um
n0 = rng.uniform(1.6, 2.4, 20)                    # 20 indices at 0.55 um

J, dJ_dt, dJ_dn = ft.merit_and_grad(lam, t, n0, S, w, c0, n_sub=nsub)

def merit(tt):
    T = ft.transmittance(lam, tt, n0[:, None] * S, 1.0, nsub)
    return float(ft.merit(T, w, c0))

h = 1e-7
e = np.zeros(20); e[4] = h
fd = (merit(t + e) - merit(t - e)) / (2 * h)
print(f"merit J = {J:.6f}")
print(f"dJ/dt_5: adjoint {dJ_dt[4]:.8f}, finite difference {fd:.8f}")
```

```
merit J = 0.484024
dJ/dt_5: adjoint 1.39661451, finite difference 1.39661451
```

`notch_weights(lam_um, center_um, half_um, guard_um)` blocks
`[center_um - half_um, center_um + half_um]` and passes everything
outside a further `guard_um` on each side; wavelengths in the guard do
not count. One call to `merit_and_grad` gives all 40 derivatives; the
finite difference above needs two full calculations per derivative.

### 3. From a nominal design to a robust one

```python
import numpy as np
import fabtwin as ft

lam = np.linspace(0.45, 0.65, 41)                 # um
S = ft.dispersion_shape(lam, 0.550)
nsub = ft.SIO2_MALITSON1965.n(lam)
w, c0 = ft.notch_weights(lam, 0.532, 0.015, 0.030)  # block 532 nm
box = ft.DesignBox(8, 0.020, 0.120, 1.6, 2.4)     # 8 layers

# 1. Best nominal design (as if fabrication were perfect).
t, n0, J, _ = ft.inverse_design(lam, S, w, c0, box, n_probe=60,
                                n_seed=2, n_iter=30, n_sub=nsub, seed=0)
print(f"nominal merit: {J:.4f}")

# 2. Historical runs of a tool (here: the built-in reference process).
rng = np.random.default_rng(1)
rt, rn, ftd, fnd = ft.PAPER_PROCESS.trace_dataset(*box.sample(rng, 40), 2, rng)
twin = ft.GaussianTwin(ft.errors_from_traces(rt, rn, ftd, fnd))

# 3. Make the worst 10 % of fabricated outcomes as good as possible.
t_r, n_r, _ = ft.robustify(lam, t, n0, S, w, c0, twin, box, alpha=0.1,
                           K=32, steps=40, lr=3e-3, seed=0, n_sub=nsub)

# 4. Score both designs with 500 fresh runs of the reference process.
def score(tt, nn):
    r = np.random.default_rng(7)
    return ft.evaluate_under_process(
        lambda a, b, K: ft.PAPER_PROCESS.ensemble(a, b, K, r),
        tt, nn, lam, S, w, c0, K=500, n_sub=nsub, alpha=0.1)

for name, (tt, nn) in [("nominal", (t, n0)), ("robust", (t_r, n_r))]:
    s = score(tt, nn)
    print(f"{name:8s} mean {s['mean']:.4f}  CVaR(10 %) {s['CVaR']:.4f}")
```

```
nominal merit: 0.7704
nominal  mean 0.7335  CVaR(10 %) 0.6779
robust   mean 0.7583  CVaR(10 %) 0.7218
```

In this run the robust design keeps its worst 10 % of fabricated
devices better than the nominal design does (0.7218 against 0.6779),
when both are scored by the reference process rather than by the twin
that was used to robustify. That is one illustrative case, not a
guarantee: the tests check only that robustification raises the
CVaR under the twin it was given (see below).

`inverse_design` first scores `n_probe` random designs, then refines
the best `n_seed` of them by gradient ascent (Adam, a standard rule
for choosing the step sizes; each step is pulled back inside the
box). `robustify` draws `K` fresh samples of fabrication errors from
the twin at each of `steps` steps and follows the exact gradient of
their CVaR (or of mean minus `beta` times standard deviation with
`mean_variance=True`). Short runs are used here to keep the example
fast; the defaults are `K=128`, `steps=150`.

### 4. Your own run logs: the trace file

```python
import os, tempfile
import numpy as np
import fabtwin as ft

rng = np.random.default_rng(0)
box = ft.DesignBox(4, 0.040, 0.120, 1.7, 2.1)
rt, rn, ftd, fnd = ft.PAPER_PROCESS.trace_dataset(*box.sample(rng, 3), 2, rng)

path = os.path.join(tempfile.mkdtemp(), "fab_traces.csv")
ft.save_traces_csv(path, rt, rn, ftd, fnd)
print(open(path).readline().strip())              # the header

rt, rn, ftd, fnd = ft.load_traces_csv(path)
ftd[2, 1] *= 1000.0                               # a value typed in nm, not um
report = ft.validate_traces(rt, rn, ftd, fnd)
print(report["n_runs"], "runs,", report["n_layers"], "layers")
for run, layer, kind, value in report["flags"]:
    print(f"flag: run index {run}, layer index {layer}, {kind} = {value:.1f}")
```

```
run,layer,t_recipe_um,n_recipe,t_fab_um,n_fab
6 runs, 4 layers
flag: run index 2, layer index 1, rel_t = 1041.1
```

The file has one row per run and layer: thicknesses in micrometres,
indices at the reference wavelength, and layers numbered 1 to `N`
with no gaps in every run. `validate_traces` raises on structural
problems (wrong shapes, non-finite values, non-positive thicknesses)
but only **flags** implausible values (a relative thickness error or
an index error larger than 0.5 in size, by default; both limits are
keyword arguments). It
does not drop them: unusual runs are exactly what a twin should learn
from, so the decision is yours. Flag indices count from 0.

If your tool logs thickness but not index, write `n_fab = n_recipe`;
the fitted twin then has almost no index errors (the test asserts
below 1e-3), and `DesignBox.thickness_only` freezes the indices during
design.

To judge a twin on real data, fit it on part of the runs and score it
on the rest with `twin_fidelity_report`: it compares means,
covariances and correlations of the error vectors, and the mean,
lower percentile, CVaR and Wasserstein distance (W1) of the merit
distributions the two sets of errors produce on a set of designs,
through the exact optics. W1 measures how far apart two sets of
values are as a whole: roughly, the average distance each value must
be moved to turn one set into the other (0 when they are identical).

### 5. Thickness errors from one measured spectrum

```python
import numpy as np
import fabtwin as ft

lam = np.linspace(0.40, 0.80, 201)
S = ft.dispersion_shape(lam, 0.550)
nsub = ft.SIO2_MALITSON1965.n(lam)
recipe_t = np.random.default_rng(1).uniform(0.050, 0.120, 6)     # um
n0 = np.array([1.46, 2.0, 1.46, 2.0, 1.46, 2.0])

# Pretend the tool made these relative thickness errors, and all we
# have is the measured transmittance, with 0.2 % measurement noise.
x_true = np.array([0.015, -0.022, 0.030, -0.011, 0.006, -0.028])
T, _, _ = ft.transmittance_and_grads(lam, recipe_t * (1 + x_true), n0, S,
                                     n_sub=nsub)
T_meas = np.clip(T + np.random.default_rng(7).normal(0, 2e-3, lam.size), 0, 1)

rec = ft.errors_from_spectrum(lam, T_meas, recipe_t, n0, S, n_sub=nsub,
                              sigma_T=2e-3)
for i in range(6):
    print(f"layer {i + 1}: true {x_true[i]:+.4f}  "
          f"recovered {rec.dt_over_t[i]:+.4f} +/- {rec.sigma[i]:.4f}")
print(f"chi2 {rec.chi2:.1f} for {rec.dof} degrees of freedom")

# Two touching layers of the same index: only their total thickness shows.
t3, n3 = np.array([0.080, 0.060, 0.100]), np.array([2.0, 2.0, 1.46])
T3, _, _ = ft.transmittance_and_grads(lam, t3, n3, S, n_sub=nsub)
try:
    ft.errors_from_spectrum(lam, T3, t3, n3, S, n_sub=nsub)
except ValueError as err:
    print("refused:", str(err).split(":")[0])
```

```
layer 1: true +0.0150  recovered +0.0217 +/- 0.0027
layer 2: true -0.0220  recovered -0.0204 +/- 0.0012
layer 3: true +0.0300  recovered -0.0182 +/- 0.0206
layer 4: true -0.0110  recovered -0.0107 +/- 0.0004
layer 5: true +0.0060  recovered +0.0412 +/- 0.0155
layer 6: true -0.0280  recovered -0.0350 +/- 0.0036
chi2 146.0 for 195 degrees of freedom
refused: non-unique recovery
```

This is the same case as one of the tests. `chi2` is the sum of the
squared misfits, each divided by the stated noise level; when the fit
is good and the noise level is right it is close to the number of
degrees of freedom (here, wavelengths minus layers). The uncertainties
matter:
layers 3 and 5 are poorly determined by this spectrum (their `+/-` is
large), and the recovered values there are far from the truth; the
well-determined layers are close. `errors_from_spectrum` fits
thickness errors only, holding the indices at the recipe values. It
tries `n_starts` starting points (8 by default) and refuses when
different error vectors fit the spectrum equally well, when some
combination of layers has no effect on the spectrum, or when there
are fewer wavelengths than layers.

### 6. Planning the calibration runs

```python
import numpy as np
import fabtwin as ft

box = ft.DesignBox(6, 0.04, 0.16, 1.7, 2.1)
rec_t, rec_n = ft.design_recipes(box, n_recipes=4, seed=0)
print("first recipe, thicknesses (um):", np.round(rec_t[0], 3))

# A pilot of 64 runs of one recipe (here from the reference process).
rng = np.random.default_rng(5)
rt, rn, ftd, fnd = ft.PAPER_PROCESS.trace_dataset(
    [np.full(6, 0.08)], [np.full(6, 1.9)], 64, rng)
pilot = ft.errors_from_traces(rt, rn, ftd, fnd)
M, se = ft.runs_for_twin_mean(0.004, pilot)
print(f"runs needed for a standard error of 0.004: {M} "
      f"(largest predicted standard error {se.max():.5f})")
```

```
first recipe, thicknesses (um): [0.087 0.087 0.084 0.1   0.125 0.093]
runs needed for a standard error of 0.004: 102 (largest predicted standard error 0.00399)
```

`design_recipes` spreads recipes over the box: it draws 512 candidate
recipes, starts from the one nearest their centre, and then
repeatedly adds the candidate farthest from all recipes chosen so far
(a greedy "maximin" rule; Johnson, Moore and Ylvisaker, J. Statist.
Plann. Inference 26, 131 (1990)). Distances are measured after
scaling each range to 0..1, and frozen parameters do not count. It is
a sensible rule of thumb, not a proof of the best possible choice.
`runs_for_twin_mean` uses the standard result that the standard error
of a mean after `M` independent runs is `sqrt(variance / M)`, with the
variances estimated from a pilot of at least 8 runs, and returns the
smallest `M` that meets the target for every component.

### 7. An error band that does not trust the twin

```python
import numpy as np
import fabtwin as ft

rng = np.random.default_rng(23)
t0, n0 = np.full(4, 0.08), np.full(4, 1.9)

# 60 held-out runs; the "prediction" is simply their mean error vector.
rt, rn, ftd, fnd = ft.PAPER_PROCESS.trace_dataset([t0], [n0], 60, rng)
x = ft.errors_from_traces(rt, rn, ftd, fnd)
pred = x.mean(axis=0)
scores = np.max(np.abs(x - pred), axis=1)          # worst component per run

q = ft.conformal_quantile(scores, alpha=0.1)       # 90 % level
lo, hi = ft.conformal_interval(pred, q)
print(f"q = {q:.4f}; guaranteed coverage "
      f"{ft.conformal_coverage_exact(60, 0.1):.4f} (for continuous scores)")

# Check on 400 fresh runs.
rt, rn, ftd, fnd = ft.PAPER_PROCESS.trace_dataset([t0], [n0], 400, rng)
x_new = ft.errors_from_traces(rt, rn, ftd, fnd)
inside = np.all((x_new >= lo) & (x_new <= hi), axis=1)
print(f"fresh runs inside the interval: {inside.mean():.3f}")

try:
    ft.conformal_quantile(scores[:5], alpha=0.1)
except ValueError as err:
    print("refused:", err)
```

```
q = 0.0872; guaranteed coverage 0.9016 (for continuous scores)
fresh runs inside the interval: 0.907
refused: 5 held-out scores cannot certify level 0.9: the required rank 6 exceeds n. Hold out at least 9 runs, or lower the confidence
```

`conformal_quantile` returns the `ceil((n + 1)(1 - alpha))`-th smallest
of the `n` held-out scores. The split-conformal theorem (Vovk,
Gammerman and Shafer (2005); Lei et al., J. Am. Stat. Assoc. 113,
1094 (2018); Angelopoulos and Bates, arXiv:2107.07511) says a new
exchangeable run then falls within `q` of its prediction with
probability at least `1 - alpha`; for continuous scores the exact
probability is `ceil((n + 1)(1 - alpha)) / (n + 1)`, which
`conformal_coverage_exact` returns. This holds on average over runs,
not for each recipe separately, and only while the process does not
change.

### 8. The JAX solver and the hand adjoint agree (needs `[twin]`)

```python
import numpy as np
import jax
import jax.numpy as jnp
import fabtwin as ft
from fabtwin import twin_jax as tj                 # needs fabtwin[twin]

lam = np.linspace(0.40, 0.80, 161)
S = ft.dispersion_shape(lam, 0.550)
nsub = ft.SIO2_MALITSON1965.n(lam)
w, c0 = ft.notch_weights(lam, 0.532, 0.015, 0.030)
rng = np.random.default_rng(1)
t, n0 = rng.uniform(0.020, 0.120, 12), rng.uniform(1.6, 2.4, 12)

g_jax = jax.grad(lambda tt, nn: tj.merit_jax(lam, tt, nn, S, w, c0,
                                             n_sub=jnp.asarray(nsub)),
                 argnums=(0, 1))(jnp.asarray(t), jnp.asarray(n0))
_, g_t, g_n = ft.merit_and_grad(lam, t, n0, S, w, c0, n_sub=nsub)
diff = max(np.abs(np.asarray(g_jax[0]) - g_t).max(),
           np.abs(np.asarray(g_jax[1]) - g_n).max())
print("JAX autodiff and hand adjoint agree to 1e-12:", diff < 1e-12)
```

```
JAX autodiff and hand adjoint agree to 1e-12: True
```

Two separate derivations of the same gradient -- algebra by hand in
NumPy, automatic differentiation in JAX -- give the same numbers. The
learned twin itself is used as: build a `FabTwinConfig`, normalize the
recipes with `norm_recipe`, train with `train_wgan` on the error
vectors, draw fabricated stacks with `sample_twin`, and robustify with
`robustify_gan`. Training takes 4000 steps by default and is not
shown here.

### 9. Materials: cited formulas and your own measured tables

```python
import numpy as np
import fabtwin as ft

print(f"Si3N4 at 0.532 um: n = {ft.SI3N4_LUKE2015.n(0.532):.4f}")
print(f"fused silica at 0.532 um: n = {ft.SIO2_MALITSON1965.n(0.532):.4f}")

# Your own measured table (these numbers are illustrative only).
film = ft.TabulatedMaterial(
    name="my SiNx film",
    lam_um=[0.40, 0.50, 0.60, 0.70, 0.80],
    n_table=[2.08, 2.04, 2.02, 2.01, 2.00],
    reference="ellipsometry run 17, lab notebook p. 42 (illustrative)")
print(f"table at 0.55 um: n = {film.n(0.55):.4f}")

try:
    film.n(0.90)                                  # outside the table
except ValueError as err:
    print("refused:", str(err).split(";")[0])
```

```
Si3N4 at 0.532 um: n = 2.0559
fused silica at 0.532 um: n = 1.4607
table at 0.55 um: n = 2.0283
refused: my SiNx film: wavelength outside the tabulated range [0.4, 0.8] um (ellipsometry run 17, lab notebook p. 42 (illustrative))
```

A table is interpolated with PCHIP, a piecewise cubic that does not
overshoot between measured points. The `reference` field is required.
A table may include an extinction column `k_table`; then `.n()`
refuses (the design path is lossless) and `.nk()` gives `n + i k` for
the forward optics.

## What is in the package

Every name below is exported from `fabtwin` unless marked
`fabtwin.twin_jax`. Each docstring (`help(fabtwin.robustify)`, for
example) gives the inputs, units and conventions.

**Materials** (`fabtwin.materials`)

- `SellmeierMaterial(name, terms, lam_min, lam_max, reference)` -- a
  Sellmeier formula with its valid range and a source; `sellmeier` is
  the bare formula without a range check.
- `SI3N4_LUKE2015` -- silicon nitride, K. Luke et al., Opt. Lett. 40,
  4823 (2015), valid 0.310-5.504 um; `SIO2_MALITSON1965` -- fused
  silica, I. H. Malitson, J. Opt. Soc. Am. 55, 1205 (1965), valid
  0.21-6.7 um. Both via the refractiveindex.info database (CC0).
- `TabulatedMaterial` -- your measured table (example 9).
- `dispersion_shape(lam, lam0, material)` -- `S(lam) = n(lam)/n(lam0)`;
  `layer_index(n_at_lam0, lam, lam0, material)` -- `n0 * S(lam)`. This
  "one material, variable index" layer model follows the SiNx platform
  of Yesilyurt et al., Nanophotonics 12, 993 (2023).

**Optics** (`fabtwin.tmm`)

- `stack_rt`, `transmittance`, `reflectance` -- exact transfer-matrix
  `R` and `T` for any stack, normal or oblique incidence, s or p
  polarization, absorbing layers allowed.
- `notch_weights`, `bandpass_weights` -- the two filter merits of the
  paper as weights `(w, const)`; `merit(T, w, const)` evaluates
  `w . T + const`.
- `weights_from_reflectance(w_R, const)` -- turns a merit written in
  terms of `R` (for a mirror) into one in terms of `T`, using
  `R = 1 - T`, which holds only for non-absorbing stacks.

**Gradients** (`fabtwin.adjoint`)

- `transmittance_and_grads(lam, t, n0, shape, ...)` -- `T` and its
  derivatives with respect to every thickness and index.
  `merit_and_grad` -- `J` and its derivatives. `shape` is one shared
  dispersion shape `(L,)` or one per layer `(N, L)` (for stacks of
  two or more materials). Normal incidence and non-absorbing layers
  only.

**Fabrication process and twins** (`fabtwin.process`, `fabtwin.twins`)

- `DepositionProcess` -- a simulated deposition machine with six error
  mechanisms: a systematic thickness bias, an index drift through the
  stack, intermixing with the previous layer, correlated thickness
  noise that grows with layer thickness, skewed index noise, and rare
  particle defects that thicken one layer. Each size is a field you
  can change or set to zero. `PAPER_PROCESS` has the paper's values
  (field `reference`: Mahim et al., IEEE Sensors J. (2026), Sec. V-D;
  magnitudes per Yesilyurt 2023 / Wilbrandt 2008). Methods:
  `corrupt` (one run), `ensemble` (many runs of one recipe),
  `trace_dataset` (traces for many recipes), `deterministic_map` (the
  result with all randomness off).
- `errors_from_traces(rt, rn, ft, fn)` -> error vectors;
  `apply_errors(t, n, x)` does the reverse.
- `GaussianTwin(x, diagonal=False)` -- a normal distribution fitted to
  error vectors; `sample_errors`, `sample`, and `transform(z)`
  (`mu + L z`: turns standard normal random numbers `z` into error
  vectors, where `mu` is the mean and `L L^T` the covariance; this
  lets `robustify` hold the random numbers fixed while it takes
  derivatives).

**Risk and yield** (`fabtwin.risk`)

- `cvar(values, alpha)` -- mean of the `ceil(alpha K)` smallest values
  (Rockafellar and Uryasev, J. Risk 2, 21 (2000), lower tail).
- `tail_statistics` -- mean, standard deviation, `100*alpha`
  percentile and CVaR.
- `pass_fail(T, stop_mask, pass_mask, leak_max=0.09, pass_min=0.90)`
  -- passes when the largest `T` in the stop band is at most
  `leak_max` and the mean `T` in the pass band is at least `pass_min`;
  `yield_fraction` -- the fraction that pass.
- `evaluate_under_process(sampler, ...)` -- scores a design with `K`
  fresh fabricated samples from any process or twin.

**Design and robust design** (`fabtwin.design`, `fabtwin.robust`)

- `DesignBox(n_layers, t_lo, t_hi, n_lo, n_hi)` -- each bound a
  number or one value per layer; equal bounds freeze a parameter;
  `DesignBox.thickness_only(n_layers, t_lo, t_hi, n_fixed)`;
  `sample`, `clip`.
- `inverse_design` -- random probes then gradient ascent (query budget
  `n_probe + 2 * n_seed * n_iter`); `adam_ascent` -- the ascent from
  one start; `random_search` -- the equal-budget random baseline.
- `robustify` -- CVaR (or mean minus `beta` standard deviations)
  ascent through a `GaussianTwin` and the exact optics;
  `cvar_objective_and_grad` -- that objective and its exact gradient
  for a fixed batch of random draws. Fabricated thicknesses are
  clipped to `[0.5 t_lo, 2 t_hi]` and indices to
  `[n_lo - 0.2, n_hi + 0.2]` inside the objective.

**Your data and how good the twin is** (`fabtwin.data`,
`fabtwin.fidelity`)

- `save_traces_csv`, `load_traces_csv`, `validate_traces` (example 4).
- `moment_errors` -- largest difference of the means, and the size
  (Frobenius norm: square root of the sum of squared entries) of the
  difference of the covariance matrices and of the correlation
  matrices, between two sets of error vectors.
- `distribution_distances` -- differences in mean, lower percentile
  and CVaR between two sets of merit values, and the 1-D Wasserstein
  distance (from SciPy).
- `induced_merits` -- the merit of one design under each error vector.
- `twin_fidelity_report` -- `moment_errors` on the error vectors,
  plus the `distribution_distances` of the induced merits, averaged
  over a set of designs.

**Reverse engineering** (`fabtwin.reverse`)

- `errors_from_spectrum` -> `SpectrumRecovery` (fields `dt_over_t`,
  `t_um`, `sigma`, `chi2`, `dof`, `condition_number`, `n_converged`)
  (example 5).

**Planning** (`fabtwin.lab`)

- `design_recipes`, `runs_for_twin_mean` (example 6).

**Conformal prediction** (`fabtwin.conformal`)

- `conformal_quantile`, `conformal_interval`,
  `conformal_coverage_exact` (example 7).

**Learned twin** (`fabtwin.twin_jax`, needs `[twin]`)

- `FabTwinConfig` -- layer count, box (plain numbers, strictly
  increasing), latent size (the number of random inputs the network
  turns into one error vector), hidden-layer sizes, and `out_scale`, the
  largest error the generator can output (0.30 by default).
- `train_wgan` -- trains the conditional WGAN-GP (Gulrajani et al.,
  NeurIPS 2017) with an extra penalty matching means and covariances;
  the optional `tail` argument adds the paper's physics-in-the-loop
  calibration of the worst merit losses. The module docstring reports
  the paper's finding that this calibration helps only when traces
  are scarce.
- `sample_twin`, `robustify_gan` (CVaR ascent through the generator
  and the JAX optics together; the paper's Stage 2).
- `transmittance_jax`, `merit_jax` -- the optics in JAX, normal
  incidence; `norm_recipe`, `init_models`, `gen_forward` -- building
  blocks; `HAVE_JAX` -- whether JAX could be imported.

## When it refuses, and why

`fabtwin` raises an error instead of guessing when:

- a wavelength is outside a Sellmeier formula's valid range or a
  table's measured range;
- a measured table has no `reference`, is not strictly increasing, has
  non-positive `n` or negative `k`, or is asked for `.n()` while it
  has non-zero `k`;
- an index has a negative imaginary part (gain, not absorption);
- the gradient functions (and so `inverse_design`, `robustify`,
  `errors_from_spectrum`) get a complex thickness, index, dispersion
  shape, substrate index or incidence index -- they cover
  non-absorbing stacks only (the shape, substrate and incidence cases
  are new in 0.6.1);
- array shapes do not match (layer indices vs thicknesses and
  wavelengths, a per-layer shape that is not `(N, L)`, error vectors
  not of length `2N`);
- a notch or bandpass layout leaves no wavelengths in a band;
- a design box is inverted, has a non-positive thickness bound, or is
  fully frozen (nothing to design);
- a `FabTwinConfig` gets frozen or per-layer bounds (its recipe
  normalization divides by `hi - lo`);
- process parameters are out of range (`rho` outside [0, 1), negative
  noise, `p_flake` not a probability);
- a trace file has the wrong header, no rows, a wrong number of
  fields, a duplicate or missing layer, or runs with different layer
  counts; or trace arrays have different shapes, non-finite values or
  non-positive thicknesses;
- `errors_from_spectrum` has fewer wavelengths than layers, a measured
  `T` outside [0, 1], no converged fit, a combination of layers the
  spectrum cannot see, or two different answers that fit equally well;
- `runs_for_twin_mean` gets fewer than 8 pilot runs, a pilot with zero
  variance in every component, or a target that is not positive;
  `design_recipes` gets fewer candidates than recipes;
- a conformal level cannot be certified with the number of held-out
  scores (the message names the minimum), or scores are negative;
- `cvar` gets no values or `alpha` outside (0, 1]; `pass_fail` gets an
  empty stop or pass mask; `GaussianTwin` gets fewer than 2 error
  vectors; `twin_fidelity_report` gets no designs;
- `errors_from_spectrum` gets a `sigma_T` that is not positive and
  finite; `weights_from_reflectance` gets weights that are not 1-D;
  `pol` is not `"s"` or `"p"`;
- the `twin_jax` functions are called without JAX installed
  (`ImportError` naming the extra).

## How the results are checked

78 automated tests run on every change, on Python 3.10, 3.11, 3.12,
3.13 and 3.14 with the `[test,twin]` extras, and once more on Python
3.10 with the oldest versions `pyproject.toml` allows (NumPy 1.26.0,
SciPy 1.11.0, JAX 0.4.30, optax 0.2.0; tmm 0.1.8 for the reference
check). Without JAX, 8 of the tests do not run (pytest reports the 6
in `test_twin_jax.py` as one skipped module, plus 2 others). Numerical
checks compare against a closed-form answer, an independent code, or
a second calculation; none compares against a number stored from an
earlier run. Checks of random processes use fixed seeds and
statistical tolerances. The main checks:

**Optics**

- A bare interface (substrate with no layers) gives exactly the
  textbook Fresnel transmittance `4 n_sub / (1 + n_sub)^2`; a
  half-wave layer is invisible and a quarter-wave anti-reflection
  layer gives `T = 1`, both to 1e-14.
- `R + T = 1` to 1e-12 on random 20-layer non-absorbing stacks;
  absorbing layers absorb (`R + T < 1`); gain is refused.
- At 40 degrees, s and p reflectance match the Fresnel formulas to
  1e-14; at the Brewster angle (the angle at which p-polarized light
  is not reflected at all) p reflectance is below 1e-30; at normal
  incidence s and p transmittance are identical.
- Agreement with the independent `tmm` package (S. J. Byrnes,
  arXiv:1603.02720) to 1e-12 on random 8-layer stacks.
- `(HL)^p` quarter-wave mirrors match the textbook closed form
  `R = ((1 - Y)/(1 + Y))^2`, `Y = (nH/nL)^(2p) n_sub`, to 1e-12 for
  p = 1, 3, 6, also through `weights_from_reflectance`.
- For absorbing stacks, `weights_from_reflectance` is off by exactly
  `-w_R . A` (to 1e-12), where `A` is the absorbed fraction.
- The Sellmeier engine equals the formula evaluated directly (no
  difference); a table sampled from the Malitson formula reproduces
  its nodes to 1e-15 and the formula between nodes to 1e-6.

**Gradients**

- The adjoint's `T` equals the optics module's to 1e-13.
- Against central finite differences on a random 20-layer stack: median
  relative error below 1e-8, largest below 1e-6 (the paper reports a
  median near 1e-10 for its JAX version). With per-layer dispersion on
  a two-material stack: median below 1e-7, largest below 1e-5. With a
  tabulated material and a user-defined Sellmeier material, each
  derivative checked has a relative error below 1e-6.
- At the quarter-wave anti-reflection optimum both derivatives are
  below 1e-12.
- JAX automatic differentiation equals the hand adjoint to 1e-12, and
  the JAX optics equal the NumPy optics to 1e-12.
- Complex dispersion shapes, substrate or incidence indices are
  refused (new in 0.6.1).

**Process and twins**

- With the random parts off, the reference process equals its closed
  form exactly.
- Over 4000 runs, the thickness noise has standard deviation within
  0.002 of `sig_t` and neighbour correlation within 0.05 of `rho`;
  over 20000 runs, the mean index is within 2e-3 of the noise-free
  value and the index noise is right-skewed, and the particle-defect
  rate is within 0.006 of `p_flake`.
- Errors extracted from traces and applied back reproduce the traces
  to 1e-15.
- A `GaussianTwin` fitted to 60000 samples recovers the true mean to
  3e-4 and covariance to 5e-6.

**Design and robust design**

- On a one-layer anti-reflection problem, `inverse_design` finds the
  closed-form optimum: merit within 1e-10 of 1, thickness and index
  within 1e-6.
- On an 8-layer notch it beats random search that is given one merit
  evaluation for every probe and every gradient step of
  `inverse_design` (`n_probe + n_seed * n_iter` evaluations).
- The exact CVaR gradient of `cvar_objective_and_grad` (fixed random
  draws) matches finite differences to a relative 1e-6, as does the
  mean-minus-deviation version.
- `robustify` raises the CVaR measured under the twin it used; on the
  two-material platform and with frozen indices, it does not lower it
  (tolerance 1e-6).
- With frozen indices, `inverse_design` and `robustify` return the
  frozen values unchanged (exact equality).

**Data, fidelity, reverse engineering, planning, conformal**

- The CSV trace file round-trips exactly; five kinds of malformed file
  are refused; a unit mix-up is flagged, not dropped.
- Every fidelity distance is exactly zero between a sample and itself;
  W1 equals SciPy's value exactly; a mean-shifted twin scores worse
  than a fitted one.
- Spectrum recovery without noise returns the true errors to 1e-6,
  and a perfect deposition to 1e-8. With seeded 0.2 % noise (example
  5), every recovered error is within 4 reported standard deviations
  of the truth and chi2 is between 0.5 and 1.7 times the degrees of
  freedom (one seeded case).
- Every `design_recipes` pick is re-derived independently (to 1e-12);
  the frozen index does not change the chosen thicknesses.
- `runs_for_twin_mean` returns the smallest sufficient `M` (checked on
  both sides of the target); over 60 simulated calibrations of 40
  runs, the scatter of the estimated mean matches the predicted
  standard error within 35 % for the largest component.
- `conformal_quantile` equals the rank formula exactly. In 4000
  simulated trials (n = 39, alpha = 0.1) the coverage is within 4
  standard errors of the exact value, and on the reference process
  400 fresh runs reach at least `1 - alpha` minus 4 standard errors.

**Learned twin** (`[twin]` extra): the moment penalty is exactly zero
for identical batches; short training runs finish with finite losses;
samples stay within `out_scale`; the CVaR gradient through the
generator matches finite differences to a relative 1e-5;
`robustify_gan` stays in the box. These tests check that the machinery
works, not how good a trained twin is.

## Corrections in earlier versions

**0.6.1 (this release) closes a gap in the lossless check of the
gradient functions.** `transmittance_and_grads` and `merit_and_grad`
refused complex thicknesses and indices, but a complex dispersion
shape, a complex substrate index array, or a NumPy complex scalar
substrate or incidence index was silently converted to its real part
(NumPy printed only a `ComplexWarning`), so the returned `T` and
gradients belonged to a different, non-absorbing
stack than the one given. For example, for three layers of index 2.0
and thickness 0.06 um on an absorbing substrate of index `1.5 + 0.2i`
(given as an array), 0.6.0 returned `T = 0.807` at 0.45 um where the
forward optics give `T = 0.787`. The same applied to `inverse_design`,
`robustify` and `errors_from_spectrum`, which call these functions.
This affected only calls with such complex inputs, which are outside
the documented scope; they are now refused with a clear message. As
was already the case for thicknesses and indices, a complex-typed
array is refused even when its imaginary part is zero (for example
`.nk()` of a table without `k`); pass the real values (`.n()`)
instead. If
you passed `TabulatedMaterial.nk()` values or another complex
substrate to these functions, re-run with the forward optics
(`fabtwin.tmm`) instead.

The earlier README and CHANGELOG used "exact", "exactly" or "machine
precision" for several checks that the tests perform with a small
tolerance; the list above gives the tolerances the tests actually
use. The full history is in [CHANGELOG.md](CHANGELOG.md).

## Limits

- The gradient path (adjoint, design, robust design, spectrum
  recovery) covers normal incidence and non-absorbing layers, the
  regime of the FabGAN-ID loop. Oblique and absorbing stacks have
  forward optics only; their gradients are not provided.
- Merits must be linear in `T` (`w . T + const`).
- `robustify` estimates the CVaR from `K` samples drawn fresh at each
  step. Following the paper, this gradient estimate is biased for
  finite `K` (the bias shrinks as `K` grows), and the result depends on
  the twin being a fair model of the machine.
- A held-out comparison (`twin_fidelity_report`) shows how well a twin
  matches the runs you recorded. It cannot vouch for error patterns
  the machine has never shown, and a twin needs retraining as the
  machine drifts (the paper's own caveat). The paper's yield gains
  were obtained with its simulated process.
- Spectrum recovery fits thickness errors only; indices are held at
  the recipe. Recovering thickness and index together from one
  normal-incidence spectrum is the classically unreliable problem
  (Tikhonravov and Trubetskov, Appl. Opt. 51, 245 (2012); Amotchkina
  et al., Appl. Opt. 51, 5543 (2012)). The reported uncertainties are
  a linear estimate at the best fit; example 5 shows how large they
  can be.
- The conformal guarantee is on average over runs, not per recipe,
  and needs runs from an unchanged process.
- `design_recipes` is a greedy rule of thumb, not an optimal design.
- Not included, by choice: specification-conditioned correction
  policies (the paper's Stage 3, which the paper treats as
  exploratory); neural forward surrogates (the paper's protocol study
  found the exact differentiable solver better in this setting:
  faster, exact, with exact gradients); and the paper's benchmark data
  (300 designs / 48,300 samples / 400 traces), which stays with the
  companion repository and its Zenodo archive -- `DepositionProcess`
  generates equivalent data instead.
- The programming interface may change before version 1.0.

## Where it comes from

The package is the reusable library form of the FabGAN-ID framework
of the associated paper. The paper's companion repository reproduces
the paper itself (a fixed 20-layer SiNx notch platform, with JAX
throughout); `fabtwin` generalizes it to any layer count, design box,
wavelength grid, cited material and linear merit, and replaces JAX
autodiff in the core with the hand-derived adjoint so the core needs
only NumPy and SciPy. The paper is a simulation study and names a twin
trained on real in-situ monitoring data as the essential next step;
the trace-file interface is meant for that.

T. M. Mahim, M. N. Islam, M. M. Rahman, A. S. M. Mohsin, "FabGAN-ID:
Learning the Fabrication Process for Yield-Aware Inverse Design of
Multilayer Photonic Sensor Filters", *IEEE Sensors Journal* (2026).
Companion repository:
[Learned-generative-process-twins...](https://github.com/Tanvir-Mahmud-Mahim/Learned-generative-process-twins-for-yield-aware-inverse-design-of-multilayer-photonic-sensor)
(benchmark archived on Zenodo, doi:10.5281/zenodo.21315793).

## Citing, support and license

If `fabtwin` helps your work, please cite it together with the
associated paper above. Every release is archived on Zenodo under the
concept DOI
[10.5281/zenodo.22697049](https://doi.org/10.5281/zenodo.22697049),
which always resolves to the latest version.
[CITATION.cff](CITATION.cff) has the details.

The package is written and maintained by Tanvir Mahmud Mahim
(Department of Electrical and Electronic Engineering, BRAC University),
who reviews every change and takes the final decision on scope and
releases. There is no separate governance body; design questions are
discussed in the open in issues and pull requests, and the standing
rule of [CONTRIBUTING.md](CONTRIBUTING.md) binds the maintainer exactly
as it binds contributors: a change that touches physics arrives with a
test, and a constant arrives with its source. Questions and bug
reports are welcome in the
[issue tracker](https://github.com/TaN-MM-Org/fabtwin/issues).

Licensed under Apache-2.0.
