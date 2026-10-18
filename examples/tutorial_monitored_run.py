"""Tutorial: the thickness errors of a whole monitored run.

`tutorial_monitoring.py` chooses the monitor wavelength from each
layer's error on its own, as if the layers below it were exact. In a
real run they are not: each layer is stopped when the signal of the
stack as deposited reaches a cut-off computed from the recipe, so the
errors already made move every later stop. This script

1. designs the 8-layer notch filter of README example 3,
2. scans the monitor wavelengths from 0.40 to 0.80 um with
   `monitoring_errors` (0.1 % signal noise) and compares the best
   wavelength for the whole run with the best one layer by layer,
3. checks the whole-run errors against 2000 runs simulated with
   `simulate_monitored_runs` (no linearization), at the best
   wavelength,
4. picks the wavelength that is best for the filter itself (the worst
   10 % of its merit, CVaR) from the twin `monitoring_errors(...).twin()`
   of every usable wavelength, and checks it with simulated runs, the
   same noise at every wavelength (a paired comparison),
5. shows what a 0.2 % calibration offset of the monitor does.

Everything is simulated with fixed seeds; it runs in about 10 seconds.
Printed output (fabtwin 0.11.0, Python 3.13, NumPy 2.5, SciPy 1.18):

    design merit 0.7704
    usable wavelengths: 14 layer by layer, 11 for the whole run
    best layer by layer: 0.470 um, worst relative error 0.0054 alone, 0.0123 in the whole run
    best for the whole run: 0.490 um, worst relative error 0.0066 alone, 0.0101 in the whole run
    at 0.490 um, sigma_t in nm:
    layer  alone  whole run  simulated runs
        7  0.388      0.388   0.391 +/- 0.006
        6  0.450      0.676   0.655 +/- 0.010
        5  0.139      0.410   0.393 +/- 0.006
        4  0.138      0.239   0.237 +/- 0.004
        3  0.139      0.225   0.222 +/- 0.004
        2  0.181      0.294   0.293 +/- 0.005
        1  0.139      0.310   0.313 +/- 0.005
        0  0.086      0.174   0.174 +/- 0.003
    correlation of layers 1 and 2: predicted -0.815, simulated -0.815
    best for the filter (CVaR of the twin): 0.470 um
    CVaR(10 %) at 0.470 um: simulated runs 0.770221 +/- 0.000003, twin 0.770208 +/- 0.000004
    CVaR(10 %) at 0.490 um: simulated runs 0.770165 +/- 0.000005, twin 0.770152 +/- 0.000005
    simulated runs, 0.470 um minus 0.490 um: 0.000056, 95 % interval 0.000048 to 0.000064
    offset +0.2 %: thickness shifts (nm), layers in deposition order
      predicted: [-0.78  0.11  0.65 -0.28  0.26 -0.24  0.26 -0.14]
      simulated: [-0.83  0.13  0.67 -0.28  0.26 -0.24  0.26 -0.14]
"""
import warnings

import numpy as np

import fabtwin as ft

# 1. the design (README example 3)
lam = np.linspace(0.45, 0.65, 41)
S = ft.dispersion_shape(lam, 0.550)
nsub = ft.SIO2_MALITSON1965.n(lam)
w, c0 = ft.notch_weights(lam, 0.532, 0.015, 0.030)
box = ft.DesignBox(8, 0.020, 0.120, 1.6, 2.4)
t, n0, J, _ = ft.inverse_design(lam, S, w, c0, box, n_probe=60, n_seed=2,
                                n_iter=30, n_sub=nsub, seed=0)
print(f"design merit {J:.4f}")


def at(lam_mon):
    """Indices of the layers and of the substrate at the monitor
    wavelength."""
    return (n0 * ft.dispersion_shape(lam_mon, 0.550),
            float(ft.SIO2_MALITSON1965.n(lam_mon)))


# 2. scan the monitor wavelengths, whole run against layer by layer
sigma_signal = 1e-3                                  # 0.1 % noise in T
whole, alone = {}, {}
for lam_mon in np.round(np.arange(0.40, 0.8001, 0.01), 2):
    n_mon, ns_mon = at(lam_mon)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # unusable ones warn
        p = ft.monitoring_plan(lam_mon, t, n_mon, n_sub=ns_mon,
                               sigma_signal=sigma_signal)
    if np.isfinite(p.worst_rel_sigma_t()):
        alone[lam_mon] = p.worst_rel_sigma_t()
    try:
        whole[lam_mon] = ft.monitoring_errors(lam_mon, t, n_mon, sigma_signal,
                                              n_sub=ns_mon)
    except ValueError:                               # refused: unreliable
        pass
best_alone = min(alone, key=alone.get)
best = min(whole, key=lambda k: whole[k].rel_sigma_t.max())
print(f"usable wavelengths: {len(alone)} layer by layer, {len(whole)} for "
      "the whole run")
print(f"best layer by layer: {best_alone:.3f} um, worst relative error "
      f"{alone[best_alone]:.4f} alone, "
      f"{whole[best_alone].rel_sigma_t.max():.4f} in the whole run")
print(f"best for the whole run: {best:.3f} um, worst relative error "
      f"{alone[best]:.4f} alone, {whole[best].rel_sigma_t.max():.4f} in "
      "the whole run")

# 3. check the whole-run errors with 2000 simulated runs
noise = np.random.default_rng(3).normal(0.0, sigma_signal, (2000, 8))
runs = {}
for k in (best, best_alone):
    n_mon, ns_mon = at(k)
    runs[k] = ft.simulate_monitored_runs(k, t, n_mon, noise=noise,
                                         n_sub=ns_mon)
me = whole[best]
dt = runs[best] - t
sd = dt.std(axis=0, ddof=1)
se = sd / np.sqrt(2 * (dt.shape[0] - 1))             # standard error of an sd
print(f"at {best:.3f} um, sigma_t in nm:")
print("layer  alone  whole run  simulated runs")
for j in me.order:
    print(f"{j:5d} {1e3 * me.sigma_t_independent[j]:6.3f} "
          f"{1e3 * me.sigma_t[j]:10.3f}   {1e3 * sd[j]:.3f} +/- "
          f"{1e3 * se[j]:.3f}")
corr = np.corrcoef(dt.T)
print(f"correlation of layers 1 and 2: predicted {me.correlation[1, 2]:+.3f}, "
      f"simulated {corr[1, 2]:+.3f}")

# 4. the filter: which wavelength gives the best worst 10 % of the merit?
#    First from the twin of every usable wavelength (fast, linear), then
#    checked with the simulated runs, the same noise at both wavelengths
#    (common random numbers, so the comparison is paired)
def twin_cvar(lam_mon):
    tw = whole[lam_mon].twin()
    out = ft.evaluate_under_process(
        lambda tt, nn, K: tw.sample(np.random.default_rng(4), tt, nn, K),
        t, n0, lam, S, w, c0, 2000, n_sub=nsub, alpha=0.1)
    return out["samples"]


twin_J = {k: twin_cvar(k) for k in whole}
best_f = max(twin_J, key=lambda k: ft.cvar(twin_J[k], 0.1))
print(f"best for the filter (CVaR of the twin): {best_f:.3f} um")
if best_f not in runs:
    n_mon, ns_mon = at(best_f)
    runs[best_f] = ft.simulate_monitored_runs(best_f, t, n_mon, noise=noise,
                                              n_sub=ns_mon)
J_by = {}
for k in sorted(runs):
    T = np.array([ft.transmittance(lam, tk, n0[:, None] * S, 1.0, nsub)
                  for tk in runs[k]])
    J_by[k] = ft.merit(T, w, c0)
    c = ft.cvar_interval(J_by[k], 0.1)
    ct = ft.cvar_interval(twin_J[k], 0.1)
    print(f"CVaR(10 %) at {k:.3f} um: simulated runs {c['CVaR']:.6f} +/- "
          f"{c['se']:.6f}, twin {ct['CVaR']:.6f} +/- {ct['se']:.6f}")
d = ft.cvar_difference(J_by[best_f], J_by[best], 0.1)
print(f"simulated runs, {best_f:.3f} um minus {best:.3f} um: "
      f"{d['difference']:.6f}, 95 % interval {d['lo']:.6f} to "
      f"{d['hi']:.6f}")

# 5. a calibration offset of the monitor: +0.2 % in T at every stop
n_mon, ns_mon = at(best)
mo = ft.monitoring_errors(best, t, n_mon, sigma_signal, n_sub=ns_mon,
                          signal_offset=0.002)
ro = ft.simulate_monitored_runs(best, t, n_mon, noise=np.zeros((1, 8)),
                                n_sub=ns_mon, signal_offset=0.002)[0]
print("offset +0.2 %: thickness shifts (nm), layers in deposition order")
print("  predicted:", np.round(1e3 * mo.mean_t[mo.order], 2))
print("  simulated:", np.round(1e3 * (ro - t)[mo.order], 2))
