"""Process twins on the common error parameterization.

Every twin in this package speaks one language, the error vector of
the FabGAN-ID study:

    x = (t_tilde/t - 1,  n_tilde - n)  in R^{2N}

(relative thickness error, absolute index error), extracted from
(recipe, outcome) traces by `errors_from_traces` and applied to a
recipe by `apply_errors` -- an exact round trip, asserted in the
tests. A twin is anything with `sample_errors(rng_or_z, K) -> (K, 2N)`;
the Gaussian twins here are the parametric baselines of the paper,
and the learned conditional WGAN-GP twin lives in `fabtwin.twin_jax`
(the optional [twin] extra).

`GaussianTwin` also exposes the reparameterization x(z) = mu + L z
with L the Cholesky factor, which is what makes the pathwise CVaR
robustification of `fabtwin.robust` differentiable through it -- the
chain rule of an affine map is exact and hand-written there.
"""
from __future__ import annotations

import numpy as np

__all__ = ["GaussianTwin", "apply_errors", "errors_from_traces"]


def errors_from_traces(rt, rn, ft, fn):
    """x = (ft/rt - 1, fn - rn), stacked to (M, 2N)."""
    rt = np.asarray(rt, float)
    if np.any(rt <= 0):
        raise ValueError("recipe thicknesses must be positive")
    return np.concatenate([np.asarray(ft, float) / rt - 1.0,
                           np.asarray(fn, float) - np.asarray(rn, float)],
                          axis=-1)


def apply_errors(t_um, n0, x):
    """Fabricated (t_tilde, n_tilde) from a recipe and error vectors.
    x : (..., 2N). Returns arrays broadcast to x's leading shape."""
    t = np.asarray(t_um, float)
    n = np.asarray(n0, float)
    N = t.shape[-1]
    x = np.asarray(x, float)
    if x.shape[-1] != 2 * N:
        raise ValueError("error vectors must have length 2N")
    return t * (1.0 + x[..., :N]), n + x[..., N:]


class GaussianTwin:
    """Gaussian corruption model fitted to traces (diagonal or full
    covariance) -- the parametric baseline the learned twin is scored
    against, and the differentiable reference twin of the numpy core.
    """

    def __init__(self, x_traces, diagonal=False, jitter=1e-9):
        x = np.asarray(x_traces, float)
        if x.ndim != 2 or x.shape[0] < 2:
            raise ValueError("x_traces must be (M >= 2, 2N)")
        self.dim = x.shape[1]
        self.mu = x.mean(axis=0)
        if diagonal:
            self.cov = np.diag(x.var(axis=0) + jitter)
        else:
            self.cov = np.cov(x.T) + jitter * np.eye(self.dim)
        self.L = np.linalg.cholesky(self.cov)
        self.diagonal = bool(diagonal)

    def transform(self, z):
        """Reparameterization x = mu + L z (exact, differentiable:
        dx/dz = L)."""
        z = np.asarray(z, float)
        return self.mu + z @ self.L.T

    def sample_errors(self, rng, K):
        return self.transform(rng.normal(size=(K, self.dim)))

    def sample(self, rng, t_um, n0, K):
        """K fabricated realizations of a recipe."""
        return apply_errors(t_um, n0, self.sample_errors(rng, K))

    def conditional(self, observed, values):
        """The twin given some error components already measured (new in
        0.7.0) -- for example the layers deposited so far, measured by
        an in-situ monitor.

        observed : indices (into the 2N error vector) that are known;
        values : their measured values. Returns a `GaussianTwin` of the
        same dimension whose observed components are fixed at the
        measured values and whose other components follow the exact
        Gaussian conditional distribution

            mu_u + C_uo C_oo^-1 (x_o - mu_o),
            C_uu - C_uo C_oo^-1 C_ou,

        so errors correlated with the measured ones (the AR(1) thickness
        noise of a real tool, for one) are predicted, not ignored.
        """
        obs = np.asarray(observed, dtype=int).ravel()
        val = np.asarray(values, dtype=float).ravel()
        if obs.size != val.size or obs.size == 0 or \
                np.any(obs < 0) or np.any(obs >= self.dim) or \
                np.unique(obs).size != obs.size:
            raise ValueError("observed must be distinct valid indices, "
                             "one value each")
        un = np.setdiff1d(np.arange(self.dim), obs)
        C = self.cov
        Coo = C[np.ix_(obs, obs)]
        Cuo = C[np.ix_(un, obs)]
        K = np.linalg.solve(Coo, Cuo.T).T
        mu = self.mu.copy()
        mu[obs] = val
        mu[un] = self.mu[un] + K @ (val - self.mu[obs])
        cov = np.zeros_like(C)
        if un.size:
            cov[np.ix_(un, un)] = C[np.ix_(un, un)] - K @ Cuo.T
        out = GaussianTwin.__new__(GaussianTwin)
        out.dim = self.dim
        out.mu = mu
        out.cov = 0.5 * (cov + cov.T)
        out.diagonal = False
        # square-root factor with exact zeros on the observed components
        L = np.zeros_like(C)
        if un.size:
            Cu = out.cov[np.ix_(un, un)]
            w, V = np.linalg.eigh(Cu)
            L[np.ix_(un, un)] = V * np.sqrt(np.clip(w, 0.0, None))
        out.L = L
        return out
