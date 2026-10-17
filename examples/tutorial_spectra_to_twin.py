"""Tutorial: from measured spectra of finished runs to a model of the
machine.

A lab has the recipe it sent to the machine and, for each finished run,
transmittance spectra of the coated part. It wants to know how the
machine errs: is there a systematic thickness bias, and how large is the
run-to-run scatter? This script

1. compares measurement plans before any tool time is spent
   (`recovery_precision`): one normal-incidence spectrum cannot pin down
   the middle layers, spectra at 0 and 60 degrees can;
2. simulates 24 runs of a hidden machine (the reference process with
   only its thickness errors switched on: a 2 % rate bias and correlated
   run-to-run noise) and "measures" each run with the chosen plan, 0.2 %
   noise per point, fixed seeds;
3. recovers every run's thickness errors (`errors_from_spectra`) and
   checks them against the hidden truth;
4. estimates the machine's bias per layer with standard errors, fits a
   Gaussian twin to the recovered errors, and compares the twin's
   run-to-run scatter with the hidden machine's.

It runs in under 30 seconds. Printed output (fabtwin 0.10.0, Python
3.13, NumPy 2.5, SciPy 1.18):

    T at 0 deg: predicted sigma of the thickness errors [0.0192 0.0366 0.028  0.0059 0.0014]
    T at 0 deg, 60 deg s and p: predicted sigma of the thickness errors [0.0013 0.0025 0.0021 0.0013 0.0009]
    recovered minus true, in reported standard deviations: rms 1.00 over 120 errors (1 expected), largest 2.70
    smallest chi-square p-value of the 24 fits: 0.136
    bias per layer:    [0.0169 0.0155 0.0177 0.0226 0.028 ]
      standard error:  [0.0035 0.0053 0.005  0.0063 0.0047]
    pooled thickness bias 0.0201 +/- 0.0033 (hidden machine: 0.02)
    twin scatter per layer:       [0.0174 0.0258 0.0243 0.0307 0.023 ]
    hidden machine, same 24 runs: [0.0174 0.0248 0.0246 0.0304 0.023 ]
"""
import dataclasses

import numpy as np

import fabtwin as ft

lam = np.linspace(0.40, 0.90, 121)                 # um
S = ft.dispersion_shape(lam, 0.55)
t0 = np.array([0.070, 0.095, 0.060, 0.110, 0.080])  # the recipe (um)
n0 = np.array([2.2, 1.5, 2.2, 1.5, 2.2])
N = t0.size
noise = 0.002                                       # 0.2 % in T


# 1. choose the measurement before measuring
def planned(angle_deg, pol):
    return ft.Measurement(np.deg2rad(angle_deg), pol, "T", None, noise)


plans = {"T at 0 deg": [planned(0, "s")],
         "T at 0 deg, 60 deg s and p": [planned(0, "s"), planned(60, "s"),
                                        planned(60, "p")]}
for name, ms in plans.items():
    p = ft.recovery_precision(lam, ms, t0, n0, S, n_sub=1.46)
    print(f"{name}: predicted sigma of the thickness errors "
          f"{np.round(p.sigma_t, 4)}")
plan = plans["T at 0 deg, 60 deg s and p"]

# 2. the hidden machine: thickness errors only (index errors switched off)
machine = dataclasses.replace(ft.PAPER_PROCESS, a_n=0.0, kappa=0.0,
                              sig_n=0.0, p_flake=0.0)
M = 24
tf, nf = machine.ensemble(t0, n0, M, np.random.default_rng(3))
x_true = tf / t0 - 1.0                              # hidden from the lab
rng = np.random.default_rng(4)


def measure(t_made):
    out = []
    for m in plan:
        g = ft.stack_rta_and_grads(lam, t_made, n0[:, None] * S, 1.0, 1.46,
                                   m.theta0_rad, m.pol)
        v = np.clip(g["T"] + rng.normal(0.0, noise, lam.size), 0.0, 1.0)
        out.append(ft.Measurement(m.theta0_rad, m.pol, "T", v, noise))
    return out


# 3. recover every run from its spectra
x_rec = np.empty((M, N))
sig_rec = np.empty((M, N))
pvals = np.empty(M)
for k in range(M):
    r = ft.errors_from_spectra(lam, measure(tf[k]), t0, n0, S, n_sub=1.46,
                               n_starts=4, seed=k)
    x_rec[k], sig_rec[k], pvals[k] = r.dt_over_t, r.sigma_t, r.chi2_pvalue
z = (x_rec - x_true) / sig_rec
print(f"recovered minus true, in reported standard deviations: rms "
      f"{np.sqrt(np.mean(z ** 2)):.2f} over {z.size} errors "
      f"(1 expected), largest {np.abs(z).max():.2f}")
print(f"smallest chi-square p-value of the {M} fits: {pvals.min():.3f}")

# 4. what the machine does, with standard errors
bias = x_rec.mean(axis=0)
se = x_rec.std(axis=0, ddof=1) / np.sqrt(M)
print("bias per layer:   ", np.round(bias, 4))
print("  standard error: ", np.round(se, 4))
pooled = x_rec.mean()
pooled_se = x_rec.mean(axis=1).std(ddof=1) / np.sqrt(M)
print(f"pooled thickness bias {pooled:.4f} +/- {pooled_se:.4f} "
      f"(hidden machine: {machine.beta_t})")
twin = ft.GaussianTwin(np.hstack([x_rec, np.zeros((M, N))]))
twin_sd = np.sqrt(np.diag(twin.cov)[:N])
print("twin scatter per layer:      ", np.round(twin_sd, 4))
print("hidden machine, same 24 runs:", np.round(x_true.std(axis=0, ddof=1),
                                                4))
