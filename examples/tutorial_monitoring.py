"""Tutorial: choose the monitor wavelength for a coating run.

A deposition machine with an optical monitor stops each layer when the
monitor signal (here the transmittance at one wavelength) reaches a
value computed from the recipe. Near a maximum or minimum of the signal
(a turning point) the signal hardly changes with thickness, so its noise
turns into large thickness errors. This script

1. designs an 8-layer notch filter (as in README example 3),
2. plans level monitoring at every candidate wavelength from 0.40 to
   0.80 um with `monitoring_plan`, for a signal noise of 0.1 % in T,
   and picks the wavelength with the smallest worst-layer relative
   thickness error,
3. checks the predicted thickness errors against 400 simulated noisy
   stops per layer (each stop found by root finding on the forward
   optics, independently of the planner),
4. shows what the choice means for the filter: the worst 10 % of the
   merit (CVaR) under the predicted thickness errors, best wavelength
   against the worst usable one, with confidence intervals.

Everything is simulated with fixed seeds; it runs in under 30 seconds.
Printed output (fabtwin 0.10.0, Python 3.13, NumPy 2.5, SciPy 1.18):

    design merit 0.7704
    wavelengths that can monitor every layer: 14 of 41
    best monitor wavelength 0.470 um: worst relative thickness error 0.0054
    worst usable wavelength 0.730 um: worst relative thickness error 0.0506
    layer  turning points  predicted sigma_t (nm)  simulated (nm)
        7               1                   0.265   0.256 +/- 0.009
        6               2                   0.206   0.209 +/- 0.007
        5               1                   0.167   0.172 +/- 0.006
        4               1                   0.087   0.086 +/- 0.003
        3               1                   0.154   0.156 +/- 0.006
        2               2                   0.306   0.310 +/- 0.011
        1               0                   0.164   0.170 +/- 0.006
        0               1                   0.059   0.058 +/- 0.002
    CVaR(10 %) of the merit at 0.470 um: 0.76998 +/- 0.00001
    CVaR(10 %) of the merit at 0.730 um: 0.76330 +/- 0.00025
    difference 0.00668, 95 % interval 0.00621 to 0.00715
"""
import warnings

import numpy as np
from scipy.optimize import brentq

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

# 2. plan level monitoring at every candidate wavelength
sigma_signal = 1e-3                        # 0.1 % noise in T
plans = {}
for lam_mon in np.round(np.arange(0.40, 0.8001, 0.01), 2):
    n_mon = n0 * ft.dispersion_shape(lam_mon, 0.550)   # indices at lam_mon
    ns_mon = float(ft.SIO2_MALITSON1965.n(lam_mon))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")    # unusable wavelengths warn
        plans[lam_mon] = ft.monitoring_plan(lam_mon, t, n_mon, n_sub=ns_mon,
                                            sigma_signal=sigma_signal)
usable = {k: p for k, p in plans.items()
          if np.isfinite(p.worst_rel_sigma_t())}
best = min(usable, key=lambda k: usable[k].worst_rel_sigma_t())
worst = max(usable, key=lambda k: usable[k].worst_rel_sigma_t())
print(f"wavelengths that can monitor every layer: {len(usable)} of "
      f"{len(plans)}")
print(f"best monitor wavelength {best:.3f} um: worst relative thickness "
      f"error {usable[best].worst_rel_sigma_t():.4f}")
print(f"worst usable wavelength {worst:.3f} um: worst relative thickness "
      f"error {usable[worst].worst_rel_sigma_t():.4f}")

# 3. check the prediction with simulated noisy stops (forward optics only)
p = usable[best]
n_mon = n0 * ft.dispersion_shape(best, 0.550)
ns_mon = float(ft.SIO2_MALITSON1965.n(best))
rng = np.random.default_rng(11)
print("layer  turning points  predicted sigma_t (nm)  simulated (nm)")
for j in p.order:
    def signal(tau, target):
        T = ft.transmittance([best], np.concatenate([[tau], t[j + 1:]]),
                             n_mon[j:], 1.0, ns_mon)[0]
        return T - target
    st = p.sigma_t[j]
    stops = [brentq(signal, t[j] - 6 * st, t[j] + 6 * st,
                    args=(p.cutoff[j] - e,), xtol=1e-13)
             for e in rng.normal(0.0, sigma_signal, 400)]
    sd = np.std(np.array(stops) - t[j], ddof=1)
    se = sd / np.sqrt(2 * (len(stops) - 1))     # standard error of an sd
    print(f"{j:5d} {p.turning_points[j]:15d} {1e3 * st:23.3f} "
          f"{1e3 * sd:7.3f} +/- {1e3 * se:.3f}")

# 4. what it means for the filter: merit under the predicted errors
z = np.random.default_rng(5).normal(size=(2000, 8))   # same draws for both
J_by = {}
for k in (best, worst):
    tt = t * (1.0 + z * usable[k].rel_sigma_t)
    T = np.array([ft.transmittance(lam, tk, n0[:, None] * S, 1.0, nsub)
                  for tk in tt])
    J_by[k] = ft.merit(T, w, c0)
cb = ft.cvar_interval(J_by[best], 0.1)
cw = ft.cvar_interval(J_by[worst], 0.1)
d = ft.cvar_difference(J_by[best], J_by[worst], 0.1)
print(f"CVaR(10 %) of the merit at {best:.3f} um: {cb['CVaR']:.5f} +/- "
      f"{cb['se']:.5f}")
print(f"CVaR(10 %) of the merit at {worst:.3f} um: {cw['CVaR']:.5f} +/- "
      f"{cw['se']:.5f}")
print(f"difference {d['difference']:.5f}, 95 % interval {d['lo']:.5f} to "
      f"{d['hi']:.5f}")
