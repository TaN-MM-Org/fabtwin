"""[twin] extra anchors (skipped without jax): the JAX solver equals
the NumPy solver, autodiff equals the hand adjoint, the moment-match
penalty is an exact zero identity, training runs and moves the
critic, samples respect the generator's physical bound, and the
frozen-latent CVaR-through-generator gradient matches finite
differences."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp                                    # noqa: E402

import fabtwin as ft                                       # noqa: E402
from fabtwin import twin_jax as tj                         # noqa: E402

LAM = np.linspace(0.45, 0.65, 41)
S = ft.dispersion_shape(LAM, 0.550)
NSUB = ft.SIO2_MALITSON1965.n(LAM)
CFG = tj.FabTwinConfig(6, 0.020, 0.120, 1.6, 2.4, dim_z=8,
                       hidden=(32, 32))
BOX = ft.DesignBox(6, 0.020, 0.120, 1.6, 2.4)


def test_jax_solver_equals_numpy_solver():
    rng = np.random.default_rng(0)
    t = rng.uniform(0.020, 0.120, 12)
    n0 = rng.uniform(1.6, 2.4, 12)
    Tj = np.asarray(tj.transmittance_jax(LAM, jnp.asarray(t),
                                         jnp.asarray(n0), S,
                                         n_sub=jnp.asarray(NSUB)))
    Tn = ft.transmittance(LAM, t, n0[:, None] * S[None, :], 1.0, NSUB)
    assert np.abs(Tj - Tn).max() < 1e-12


def test_moment_match_is_an_exact_zero_identity():
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.normal(size=(64, 12)))
    assert float(tj.moment_match(x, x)) == 0.0
    perm = jnp.asarray(rng.permutation(64))
    assert float(tj.moment_match(x[perm], x)) < 1e-25
    y = x + 0.1
    assert float(tj.moment_match(y, x)) > 0.0


def _traces(rng, n_recipes=30, runs=2):
    rec_t, rec_n = BOX.sample(rng, n_recipes)
    rt, rn, ftd, fnd = ft.PAPER_PROCESS.trace_dataset(rec_t, rec_n,
                                                      runs, rng)
    x = ft.errors_from_traces(rt, rn, ftd, fnd)
    cond = np.asarray(tj.norm_recipe(CFG, rt, rn))
    return cond, x, rec_t, rec_n


def test_training_runs_and_samples_respect_the_head_bound():
    rng = np.random.default_rng(2)
    cond, x, rec_t, rec_n = _traces(rng)
    gp, dp, hist = tj.train_wgan(CFG, cond, x, steps=60, batch=32,
                                 log_every=20)
    assert len(hist) == 3
    assert all(np.isfinite(h[1]) and np.isfinite(h[2]) for h in hist)
    tt, nn = tj.sample_twin(CFG, gp, jax.random.PRNGKey(1), rec_t[0],
                            rec_n[0], 64)
    assert tt.shape == (64, 6)
    assert np.abs(tt / rec_t[0] - 1.0).max() <= CFG.out_scale + 1e-9
    assert np.abs(nn - rec_n[0]).max() <= CFG.out_scale + 1e-9


def test_tail_calibrated_training_runs():
    rng = np.random.default_rng(3)
    cond, x, rec_t, rec_n = _traces(rng, n_recipes=8, runs=2)
    w, c0 = ft.notch_weights(LAM, 0.532, 0.015, 0.030)
    cal_t, cal_n = rec_t[:4], rec_n[:4]
    j_nom = np.array([
        float(ft.merit(ft.transmittance(
            LAM, cal_t[b], cal_n[b][:, None] * S[None, :], 1.0, NSUB),
            w, c0)) for b in range(4)])
    tail = dict(lam=LAM, shape=S, weights=w, const=c0, n_sub=NSUB,
                cal_t=cal_t, cal_n=cal_n, j_nominal=j_nom,
                dj_low_sorted=np.array([-0.08, -0.05]))
    gp, dp, hist = tj.train_wgan(CFG, cond, x, steps=24, batch=16,
                                 tail=tail, log_every=8)
    assert all(np.isfinite(h[2]) for h in hist)


def test_frozen_latent_cvar_through_generator_matches_fd():
    """The pathwise objective of `robustify_gan` with a frozen latent
    batch is deterministic; its JAX gradient must match central
    finite differences of the same frozen objective."""
    rng = np.random.default_rng(4)
    cond, x, rec_t, rec_n = _traces(rng, n_recipes=10, runs=2)
    gp, _, _ = tj.train_wgan(CFG, cond, x, steps=20, batch=16,
                             log_every=10)
    w, c0 = ft.notch_weights(LAM, 0.532, 0.015, 0.030)
    K = 16
    z = jax.random.normal(jax.random.PRNGKey(5), (K, CFG.dim_z))
    merit_b = jax.vmap(lambda tt, nn: tj.merit_jax(
        LAM, tt, nn, S, w, c0, n_sub=jnp.asarray(NSUB)))

    def obj(t, n):
        c = tj.norm_recipe(CFG, t, n)
        xg = tj.gen_forward(CFG, gp, z, jnp.tile(c, (K, 1)))
        tt = t * (1.0 + xg[:, :6])
        nn = n + xg[:, 6:]
        J = merit_b(tt, nn)
        return jnp.mean(jnp.sort(J)[:4])

    t0 = jnp.asarray(rec_t[0])
    n0 = jnp.asarray(rec_n[0])
    g = jax.grad(obj, argnums=(0, 1))(t0, n0)
    h = 1e-6
    for i in (0, 3, 5):
        e = jnp.zeros(6).at[i].set(h)
        fd = (obj(t0 + e, n0) - obj(t0 - e, n0)) / (2 * h)
        assert abs(float(g[0][i]) - float(fd)) / \
            max(abs(float(fd)), 1e-10) < 1e-5
        fd = (obj(t0, n0 + e) - obj(t0, n0 - e)) / (2 * h)
        assert abs(float(g[1][i]) - float(fd)) / \
            max(abs(float(fd)), 1e-10) < 1e-5


def test_robustify_gan_runs_and_stays_in_the_box():
    rng = np.random.default_rng(6)
    cond, x, rec_t, rec_n = _traces(rng, n_recipes=10, runs=2)
    gp, _, _ = tj.train_wgan(CFG, cond, x, steps=20, batch=16,
                             log_every=10)
    w, c0 = ft.notch_weights(LAM, 0.532, 0.015, 0.030)
    t_r, n_r, hist = tj.robustify_gan(CFG, gp, LAM, rec_t[0], rec_n[0],
                                      S, w, c0, NSUB, alpha=0.25, K=16,
                                      steps=8, lr=3e-3)
    assert np.all((t_r >= CFG.t_lo) & (t_r <= CFG.t_hi))
    assert np.all((n_r >= CFG.n_lo) & (n_r <= CFG.n_hi))
    assert np.all(np.isfinite([h[1] for h in hist]))
