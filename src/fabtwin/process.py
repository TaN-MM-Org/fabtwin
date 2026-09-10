"""Parametric virtual deposition process: the reference "hidden fab".

The FabGAN-ID benchmark scores process twins against a held-out
ground-truth simulator with six error mechanisms that are deliberately
non-Gaussian, non-independent and design-conditional (Mahim et al.,
IEEE Sensors J., 2026, Sec. V-D; magnitudes consistent with Yesilyurt
et al., Nanophotonics 12, 993 (2023) and Wilbrandt et al., Appl. Opt.
47, C49 (2008)):

 1. systematic thickness rate bias        d -> d (1 + beta_t)
 2. systematic index drift across the stack (slope a_n)
 3. inter-layer intermixing (each index pulled toward the previous
    layer's by kappa)
 4. AR(1)-correlated relative thickness noise (rho), std sig_t scaled
    by sqrt(d / d_ref) -- design-conditional
 5. right-skewed index noise: centered log-normal, scale sig_n
 6. rare particulate events: probability p_flake per run, one random
    layer thickened by N(flake_mu, flake_sig) relative

Here the process is a dataclass, so any variant -- mechanisms scaled,
re-signed, or switched off -- is one `dataclasses.replace` away; the
paper's published magnitudes are the defaults and carry the citation.
Anchors asserted in the tests rather than stated: with every random
mechanism off the corruption is an exact closed-form deterministic
map; the AR(1) noise reproduces its stationary variance; the centered
log-normal has unit mean; the flake rate matches p_flake; and the
trace dataset exposes exactly the (recipe, outcome) pairs a twin is
allowed to learn from.
"""
from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["DepositionProcess", "PAPER_PROCESS"]


@dataclasses.dataclass(frozen=True)
class DepositionProcess:
    beta_t: float = 0.02          # systematic thickness rate bias
    a_n: float = -0.04            # index drift slope across the stack
    kappa: float = 0.12           # inter-layer intermixing
    rho: float = 0.6              # AR(1) correlation of thickness noise
    sig_t: float = 0.02           # relative thickness noise std at d_ref
    sig_n: float = 0.02           # log-normal index noise scale
    p_flake: float = 0.05         # particulate probability per run
    flake_mu: float = 0.08        # particulate relative thickness error
    flake_sig: float = 0.02
    d_ref_um: float = 0.060       # noise reference thickness
    reference: str = ("Mahim et al., IEEE Sensors J. (2026), Sec. V-D; "
                      "magnitudes per Yesilyurt 2023 / Wilbrandt 2008")

    def __post_init__(self):
        if not (0.0 <= self.rho < 1.0):
            raise ValueError("rho must lie in [0, 1)")
        if self.sig_t < 0 or self.sig_n < 0 or not (0 <= self.p_flake <= 1):
            raise ValueError("noise scales must be non-negative and "
                             "p_flake a probability")
        if self.d_ref_um <= 0:
            raise ValueError("d_ref_um must be positive")

    # ------------------------------------------------------------------
    def deterministic_map(self, t_um, n0):
        """The exact corruption with every random mechanism off:
        t -> t (1 + beta_t); n -> intermixed drifted profile.
        This closed form is the anchor the sampled path is tested
        against."""
        t = np.asarray(t_um, dtype=float)
        n = np.asarray(n0, dtype=float)
        N = t.size
        grid = (np.arange(N) / (N - 1)) - 0.5 if N > 1 else np.zeros(1)
        n_d = n + self.a_n * grid
        n_prev = np.concatenate([[n_d[0]], n_d[:-1]])
        n_out = n_d + self.kappa * (n_prev - n_d)
        return t * (1.0 + self.beta_t), n_out

    def corrupt(self, t_um, n0, rng):
        """One fabricated realization (t_tilde, n_tilde)."""
        t = np.asarray(t_um, dtype=float)
        n = np.asarray(n0, dtype=float)
        N = t.size
        t_f = t * (1.0 + self.beta_t)

        scale = self.sig_t * np.sqrt(t / self.d_ref_um)
        eps = np.zeros(N)
        if self.sig_t > 0:
            innov = rng.normal(0.0, 1.0, N) * scale * \
                np.sqrt(1.0 - self.rho ** 2)
            eps[0] = rng.normal(0.0, scale[0])
            for i in range(1, N):
                eps[i] = self.rho * eps[i - 1] + innov[i]
        t_f = t_f * (1.0 + eps)

        if self.p_flake > 0 and rng.random() < self.p_flake:
            i = rng.integers(0, N)
            t_f[i] *= 1.0 + rng.normal(self.flake_mu, self.flake_sig)

        _, n_f = self.deterministic_map(t, n)
        if self.sig_n > 0:
            skew = np.exp(rng.normal(0.0, self.sig_n, N))
            n_f = n_f * skew / np.exp(self.sig_n ** 2 / 2.0)
        return t_f, n_f

    def ensemble(self, t_um, n0, K, rng):
        """K realizations; returns (K, N) thickness and index arrays."""
        D = np.empty((K, len(t_um)))
        Nn = np.empty((K, len(n0)))
        for k in range(K):
            D[k], Nn[k] = self.corrupt(t_um, n0, rng)
        return D, Nn

    def trace_dataset(self, recipes_t, recipes_n, runs_per_recipe, rng):
        """Historical (recipe, outcome) traces -- the only data a
        process twin may learn from. Returns (rt, rn, ft, fn), each
        (R * runs_per_recipe, N)."""
        rt = np.repeat(np.asarray(recipes_t, float), runs_per_recipe, 0)
        rn = np.repeat(np.asarray(recipes_n, float), runs_per_recipe, 0)
        ft = np.empty_like(rt)
        fn = np.empty_like(rn)
        for m in range(rt.shape[0]):
            ft[m], fn[m] = self.corrupt(rt[m], rn[m], rng)
        return rt, rn, ft, fn


PAPER_PROCESS = DepositionProcess()
