"""Learned generative process twin: conditional WGAN-GP (JAX extra).

The optional half of the package (install `fabtwin[twin]`): a
conditional, moment-matched Wasserstein GAN with gradient penalty
(Gulrajani et al., NeurIPS 2017) that learns the joint deposition-
error distribution x = (t_tilde/t - 1, n_tilde - n) conditioned on
the normalized recipe, from historical traces alone -- the FabGAN
twin of Mahim et al., IEEE Sensors J. (2026), generalized to any
layer count, design box, wavelength grid and linear merit. The
generator is a reparameterized sampler, so the empirical CVaR of the
fabricated merit is ascended PATHWISE through generator and solver
jointly (`robustify_gan`), and the optional physics-in-the-loop tail
calibration matches the twin's induced lower-tail merit degradations
to the real traces' on a calibration bank (`tail_w > 0`) -- the
paper's honest finding, reproduced in the docs: it helps only when
traces are scarce.

The JAX transfer-matrix solver here is asserted equal to the NumPy
solver of `fabtwin.tmm` to 1e-12, and its autodiff gradient equal to
the hand adjoint of `fabtwin.adjoint` -- two independent derivations
of the same physics agreeing is the package's core cross-validation.

Everything in this module requires jax + optax and raises a clear
ImportError otherwise; nothing else in fabtwin imports it.
"""
from __future__ import annotations

import dataclasses

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import optax
    jax.config.update("jax_enable_x64", True)
    HAVE_JAX = True
except ImportError:                                 # pragma: no cover
    HAVE_JAX = False

__all__ = ["FabTwinConfig", "transmittance_jax", "merit_jax",
           "norm_recipe", "init_models", "gen_forward", "train_wgan",
           "sample_twin", "robustify_gan", "HAVE_JAX"]


def _require_jax():
    if not HAVE_JAX:                                # pragma: no cover
        raise ImportError(
            "this module needs the [twin] extra: pip install "
            "'fabtwin[twin]' (jax + optax)")


# ---------------------- differentiable physics ----------------------


def transmittance_jax(lam_um, t_um, n0, shape, n_inc=1.0, n_sub=1.0):
    """Normal-incidence T(lam) in JAX; equals fabtwin.tmm to 1e-12
    (asserted in the tests), with exact autodiff gradients that the
    tests hold against the hand adjoint."""
    _require_jax()
    lam = jnp.asarray(lam_um)
    S = jnp.asarray(shape)
    if S.ndim == 1:
        S = jnp.broadcast_to(S[None, :], (t_um.shape[0], lam.shape[0]))
    n = n0[:, None] * S
    delta = 2.0 * jnp.pi * n * t_um[:, None] / lam[None, :]
    c = jnp.cos(delta)
    s = jnp.sin(delta)
    eta = n
    eta_sub = jnp.asarray(n_sub) + 0.0j
    if eta_sub.ndim == 0:
        eta_sub = jnp.full(lam.shape, eta_sub)

    def step(carry, mats):
        B, C = carry
        ci, si, ei = mats
        return (ci * B + 1j * si / ei * C,
                1j * ei * si * B + ci * C), None

    (B, C), _ = jax.lax.scan(step, (jnp.ones_like(eta_sub), eta_sub),
                             (c[::-1], s[::-1], eta[::-1]))
    denom = n_inc * B + C
    return 4.0 * n_inc * jnp.real(eta_sub) / jnp.abs(denom) ** 2


def merit_jax(lam_um, t_um, n0, shape, weights, const,
              n_inc=1.0, n_sub=1.0):
    _require_jax()
    T = transmittance_jax(lam_um, t_um, n0, shape, n_inc, n_sub)
    return T @ jnp.asarray(weights) + const


# ------------------------- twin definition -------------------------


@dataclasses.dataclass(frozen=True)
class FabTwinConfig:
    """Architecture and box of a conditional twin for N layers."""

    n_layers: int
    t_lo: float
    t_hi: float
    n_lo: float
    n_hi: float
    dim_z: int = 32
    hidden: tuple = (128, 128)
    out_scale: float = 0.30       # tanh head bound on |errors|

    @property
    def dim_x(self):
        return 2 * self.n_layers

    @property
    def dim_c(self):
        return 2 * self.n_layers


def norm_recipe(cfg, t, n):
    """Recipe normalized to [-1, 1] per the design box."""
    _require_jax()
    tn = (t - cfg.t_lo) / (cfg.t_hi - cfg.t_lo) * 2 - 1
    nn = (n - cfg.n_lo) / (cfg.n_hi - cfg.n_lo) * 2 - 1
    return jnp.concatenate([tn, nn], axis=-1)


def _init_mlp(key, sizes):
    params = []
    for k, (i, o) in zip(jax.random.split(key, len(sizes) - 1),
                         zip(sizes[:-1], sizes[1:])):
        w = jax.random.normal(k, (i, o)) * jnp.sqrt(2.0 / i)
        params.append((w, jnp.zeros(o)))
    return params


def _mlp(params, x):
    for w, b in params[:-1]:
        x = jax.nn.leaky_relu(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def gen_forward(cfg, gp, z, c):
    _require_jax()
    out = _mlp(gp, jnp.concatenate([z, c], axis=-1))
    return cfg.out_scale * jnp.tanh(out)


def _crit_forward(dp, x, c):
    return _mlp(dp, jnp.concatenate([x, c], axis=-1))[..., 0]


def init_models(cfg, key):
    _require_jax()
    kg, kd = jax.random.split(key)
    gp = _init_mlp(kg, (cfg.dim_z + cfg.dim_c,) + cfg.hidden
                   + (cfg.dim_x,))
    dp = _init_mlp(kd, (cfg.dim_x + cfg.dim_c,) + cfg.hidden + (1,))
    return gp, dp


# --------------------------- losses ---------------------------


def critic_loss(cfg, dp, gp, xr, c, z, key, gp_weight=10.0):
    xf = gen_forward(cfg, gp, z, c)
    lr = _crit_forward(dp, xr, c)
    lf = _crit_forward(dp, xf, c)
    eps = jax.random.uniform(key, (xr.shape[0], 1))
    xi = eps * xr + (1 - eps) * xf
    g = jax.grad(lambda x: _crit_forward(dp, x, c).sum())(xi)
    gnorm = jnp.sqrt(jnp.sum(g ** 2, axis=-1) + 1e-12)
    pen = jnp.mean((gnorm - 1.0) ** 2)
    return jnp.mean(lf) - jnp.mean(lr) + gp_weight * pen


def moment_match(xf, xr):
    """Pooled first/second-moment penalty -- exactly zero iff the
    batch means and covariances coincide (asserted in the tests)."""
    mg, mr = jnp.mean(xf, 0), jnp.mean(xr, 0)
    cg = (xf - mg).T @ (xf - mg) / xf.shape[0]
    cr = (xr - mr).T @ (xr - mr) / xr.shape[0]
    return jnp.sum((mg - mr) ** 2) + jnp.sum((cg - cr) ** 2)


def gen_loss(cfg, gp, dp, c, z, xr, mm_w=5.0):
    xf = gen_forward(cfg, gp, z, c)
    adv = -jnp.mean(_crit_forward(dp, xf, c))
    return adv + mm_w * moment_match(xf, xr)


# --------------------------- training ---------------------------


def train_wgan(cfg, cond, xreal, key=None, steps=4000, batch=128,
               n_critic=3, lr=1e-4, mm_w=5.0, tail=None, tail_w=25.0,
               log_every=200):
    """Train the conditional WGAN-GP twin on traces.

    cond : (M, dim_c) normalized recipes; xreal : (M, dim_x) error
    vectors (from `fabtwin.twins.errors_from_traces`).
    tail : optional physics-in-the-loop calibration dict with keys
        lam, shape, weights, const, n_sub, cal_t (B,N), cal_n (B,N),
        j_nominal (B,), dj_low_sorted (q,) -- the sorted lower-tail
        merit DEGRADATIONS of the calibration bank under the real
        traces. Activated after a steps//4 adversarial warmup, every
        4th generator step, per the paper.
    Returns (gp, dp, history)."""
    _require_jax()
    if key is None:
        key = jax.random.PRNGKey(0)
    key, k0 = jax.random.split(key)
    gp, dp = init_models(cfg, k0)
    optg = optax.adam(lr, b1=0.5, b2=0.9)
    optd = optax.adam(lr, b1=0.5, b2=0.9)
    sg, sd = optg.init(gp), optd.init(dp)
    cond = jnp.asarray(cond)
    xreal = jnp.asarray(xreal)
    M = cond.shape[0]

    if tail is not None:
        t_lam = jnp.asarray(tail["lam"])
        t_S = jnp.asarray(tail["shape"])
        t_w = jnp.asarray(tail["weights"])
        t_c0 = float(tail["const"])
        t_nsub = jnp.asarray(tail["n_sub"])
        cal_t = jnp.asarray(tail["cal_t"])
        cal_n = jnp.asarray(tail["cal_n"])
        j_nom = jnp.asarray(tail["j_nominal"])
        dj_low = jnp.asarray(tail["dj_low_sorted"])
        cal_c = norm_recipe(cfg, cal_t, cal_n)
        merit_b = jax.vmap(lambda tt, nn: merit_jax(
            t_lam, tt, nn, t_S, t_w, t_c0, n_sub=t_nsub))

        def tail_pen(gp, zc):
            reps = zc.shape[0] // cal_c.shape[0]
            cc = jnp.tile(cal_c, (reps, 1))
            ct = jnp.tile(cal_t, (reps, 1))
            cn = jnp.tile(cal_n, (reps, 1))
            jn = jnp.tile(j_nom, (reps,))
            x = gen_forward(cfg, gp, zc, cc)
            Nl = cfg.n_layers
            tt = jnp.clip(ct * (1.0 + x[:, :Nl]),
                          0.5 * cfg.t_lo, 2.0 * cfg.t_hi)
            nn = jnp.clip(cn + x[:, Nl:], cfg.n_lo - 0.2, cfg.n_hi + 0.2)
            dj = merit_b(tt, nn) - jn
            q = dj_low.shape[0]
            return jnp.mean((jnp.sort(dj)[:q] - dj_low) ** 2)

    @jax.jit
    def dstep(dp, sd, gp, key):
        k1, k2, k3 = jax.random.split(key, 3)
        idx = jax.random.randint(k1, (batch,), 0, M)
        z = jax.random.normal(k2, (batch, cfg.dim_z))
        l, g = jax.value_and_grad(
            lambda p: critic_loss(cfg, p, gp, xreal[idx], cond[idx],
                                  z, k3))(dp)
        up, sd = optd.update(g, sd)
        return optax.apply_updates(dp, up), sd, l

    @jax.jit
    def gstep(gp, sg, dp, key):
        k1, k2 = jax.random.split(key)
        idx = jax.random.randint(k1, (batch,), 0, M)
        z = jax.random.normal(k2, (batch, cfg.dim_z))
        l, g = jax.value_and_grad(
            lambda p: gen_loss(cfg, p, dp, cond[idx], z, xreal[idx],
                               mm_w))(gp)
        up, sg = optg.update(g, sg)
        return optax.apply_updates(gp, up), sg, l

    if tail is not None:
        @jax.jit
        def gstep_tail(gp, sg, dp, key):
            k1, k2, k3 = jax.random.split(key, 3)
            idx = jax.random.randint(k1, (batch,), 0, M)
            z = jax.random.normal(k2, (batch, cfg.dim_z))
            zc = jax.random.normal(k3, (16 * cal_c.shape[0], cfg.dim_z))
            l, g = jax.value_and_grad(
                lambda p: gen_loss(cfg, p, dp, cond[idx], z, xreal[idx],
                                   mm_w) + tail_w * tail_pen(p, zc))(gp)
            up, sg = optg.update(g, sg)
            return optax.apply_updates(gp, up), sg, l

    warmup = steps // 4
    hist = []
    for t in range(steps):
        for _ in range(n_critic):
            key, k = jax.random.split(key)
            dp, sd, dl = dstep(dp, sd, gp, k)
        key, k = jax.random.split(key)
        if tail is not None and t >= warmup and t % 4 == 0:
            gp, sg, gl = gstep_tail(gp, sg, dp, k)
        else:
            gp, sg, gl = gstep(gp, sg, dp, k)
        if t % log_every == 0:
            hist.append((t, float(dl), float(gl)))
    return gp, dp, hist


def sample_twin(cfg, gp, key, t_um, n0, K):
    """K fabricated realizations of a recipe from the twin."""
    _require_jax()
    c = norm_recipe(cfg, jnp.asarray(t_um), jnp.asarray(n0))
    z = jax.random.normal(key, (K, cfg.dim_z))
    x = np.asarray(gen_forward(cfg, gp, z, jnp.tile(c, (K, 1))))
    Nl = cfg.n_layers
    return (np.asarray(t_um) * (1.0 + x[:, :Nl]),
            np.asarray(n0) + x[:, Nl:])


# ------------------ pathwise CVaR through the twin ------------------


def robustify_gan(cfg, gp, lam, t0, n00, shape, weights, const, n_sub,
                  alpha=0.05, K=128, steps=150, lr=2e-3, seed=0,
                  mean_variance=False, beta=1.0):
    """Projected stochastic CVaR ascent through generator AND solver
    (the paper's Stage 2). Returns (t_rob, n_rob, history)."""
    _require_jax()
    lam_j = jnp.asarray(lam)
    S_j = jnp.asarray(shape)
    w_j = jnp.asarray(weights)
    nsub_j = jnp.asarray(n_sub)
    Nl = cfg.n_layers
    merit_b = jax.vmap(lambda tt, nn: merit_jax(lam_j, tt, nn, S_j, w_j,
                                                const, n_sub=nsub_j))

    def obj(dn, z):
        t, n = dn
        c = norm_recipe(cfg, t, n)
        x = gen_forward(cfg, gp, z, jnp.tile(c, (K, 1)))
        tt = jnp.clip(t * (1.0 + x[:, :Nl]), 0.5 * cfg.t_lo,
                      2.0 * cfg.t_hi)
        nn = jnp.clip(n + x[:, Nl:], cfg.n_lo - 0.2, cfg.n_hi + 0.2)
        J = merit_b(tt, nn)
        if mean_variance:
            return jnp.mean(J) - beta * jnp.std(J)
        q = max(1, int(np.ceil(alpha * K)))
        return jnp.mean(jnp.sort(J)[:q])

    vg = jax.jit(jax.value_and_grad(obj, argnums=0))
    opt = optax.adam(lr)
    dn = (jnp.asarray(t0, dtype=jnp.float64),
          jnp.asarray(n00, dtype=jnp.float64))
    st = opt.init(dn)
    key = jax.random.PRNGKey(seed)
    hist = []
    for it in range(steps):
        key, k = jax.random.split(key)
        z = jax.random.normal(k, (K, cfg.dim_z))
        v, g = vg(dn, z)
        up, st = opt.update(jax.tree.map(lambda x: -x, g), st)
        dn = optax.apply_updates(dn, up)
        dn = (jnp.clip(dn[0], cfg.t_lo, cfg.t_hi),
              jnp.clip(dn[1], cfg.n_lo, cfg.n_hi))
        if it % 10 == 0 or it == 0:
            hist.append((it, float(v)))
    return np.asarray(dn[0]), np.asarray(dn[1]), hist
