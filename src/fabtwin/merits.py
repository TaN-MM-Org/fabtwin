"""Merits that are not linear in T, measured at any angle (new in
0.7.0).

The original design path scores a stack by J = w . T + const at
normal incidence. Here a merit is any differentiable function of the
spectra R, T and A = 1 - R - T, measured under one or more conditions
(angle of incidence and polarization), and an `OpticalModel` bundles
the merit, the conditions and the fixed absorption of each layer. The
exact gradient reaches the design through `fabtwin.gradients`, so
`inverse_design`, `robustify`, `random_search` and
`evaluate_under_process` accept a model wherever they accept linear
weights.

Merits provided (each returns J and its derivatives with respect to
every spectral value; higher J is better):

* `LinearMerit(weights, const, quantity)` -- w . X + const for X =
  "T", "R" or "A" (the original merit when X = "T" at normal
  incidence);
* `TargetMerit(target, quantity, weights)` -- minus the weighted mean
  squared distance to a target spectrum;
* `SpecMarginMerit(stop_mask, pass_mask, leak_max, pass_min,
  sharpness)` -- a smooth version of the pass/fail specification of
  `fabtwin.risk.pass_fail`, built so that J > 0 guarantees the
  specification is met (a log-sum-exp is never below the maximum, and
  a smooth minimum is never above the minimum; the tests assert the
  implication on random spectra);
* `FunctionMerit(func)` -- any user function returning (J, dJ/dR,
  dJ/dT, dJ/dA).

Every merit's gradient is checked against finite differences in the
tests.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .gradients import layer_indices, stack_rta_and_grads

__all__ = ["LinearMerit", "TargetMerit", "SpecMarginMerit",
           "FunctionMerit", "OpticalModel", "model_spectra",
           "model_merit_and_grad"]

_Q = ("R", "T", "A")


def _check_quantity(q):
    if q not in _Q:
        raise ValueError("quantity must be 'R', 'T' or 'A'")
    return q


class LinearMerit:
    """J = sum over conditions and wavelengths of w * X + const."""

    def __init__(self, weights, const=0.0, quantity="T"):
        self.w = np.asarray(weights, dtype=float)
        self.const = float(const)
        self.quantity = _check_quantity(quantity)

    def __call__(self, R, T, A):
        X = dict(R=R, T=T, A=A)[self.quantity]
        w = np.broadcast_to(self.w, X.shape)
        J = float(np.sum(w * X) + self.const)
        g = {q: np.zeros_like(X) for q in _Q}
        g[self.quantity] = np.array(w, dtype=float)
        return J, g["R"], g["T"], g["A"]


class TargetMerit:
    """J = - sum w (X - target)^2 / sum w (0 for a perfect match)."""

    def __init__(self, target, quantity="T", weights=None):
        self.target = np.asarray(target, dtype=float)
        self.quantity = _check_quantity(quantity)
        self.w = None if weights is None else np.asarray(weights, float)

    def __call__(self, R, T, A):
        X = dict(R=R, T=T, A=A)[self.quantity]
        tgt = np.broadcast_to(self.target, X.shape)
        w = np.ones_like(X) if self.w is None else \
            np.broadcast_to(self.w, X.shape)
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be >= 0 and not all zero")
        d = X - tgt
        J = float(-np.sum(w * d * d) / w.sum())
        g = {q: np.zeros_like(X) for q in _Q}
        g[self.quantity] = -2.0 * w * d / w.sum()
        return J, g["R"], g["T"], g["A"]


class SpecMarginMerit:
    """Smooth pass/fail margin: J > 0 guarantees max_stop T <= leak_max
    and mean_pass T >= pass_min in every condition.

    With s = `sharpness`, per condition c
        leak_c = (1/s) log sum_stop exp(s T)    (>= max_stop T)
        m1_c = leak_max - leak_c,  m2_c = mean_pass T - pass_min
    and J = -(1/s) log sum_c [exp(-s m1_c) + exp(-s m2_c)], which is at
    most every m1_c and m2_c. Larger s tracks the true margins more
    closely (the gap is at most log(n_stop)/s + log(2 C)/s).
    """

    def __init__(self, stop_mask, pass_mask, leak_max=0.09, pass_min=0.90,
                 sharpness=200.0):
        self.stop = np.asarray(stop_mask, bool)
        self.pas = np.asarray(pass_mask, bool)
        if not self.stop.any() or not self.pas.any():
            raise ValueError("empty stop or pass mask")
        self.leak_max = float(leak_max)
        self.pass_min = float(pass_min)
        self.s = float(sharpness)
        if not self.s > 0:
            raise ValueError("sharpness must be positive")

    def margins(self, T):
        """The exact (non-smooth) margins per condition, for reporting."""
        T = np.atleast_2d(T)
        return (self.leak_max - T[:, self.stop].max(axis=1),
                T[:, self.pas].mean(axis=1) - self.pass_min)

    def __call__(self, R, T, A):
        T2 = np.atleast_2d(T)
        if T2.shape[1] != self.stop.size or self.pas.size != self.stop.size:
            raise ValueError("stop and pass masks must have one entry per "
                             "wavelength")
        s = self.s
        Ts = T2[:, self.stop]
        mx = Ts.max(axis=1, keepdims=True)
        e = np.exp(s * (Ts - mx))
        leak = mx[:, 0] + np.log(e.sum(axis=1)) / s
        p_stop = e / e.sum(axis=1, keepdims=True)        # d leak / d T
        m1 = self.leak_max - leak
        m2 = T2[:, self.pas].mean(axis=1) - self.pass_min
        m = np.concatenate([m1, m2])
        mn = m.min()
        ex = np.exp(-s * (m - mn))
        J = float(mn - np.log(ex.sum()) / s)
        wgt = ex / ex.sum()                               # dJ/dm
        C = T2.shape[0]
        gT = np.zeros_like(T2)
        gT[:, self.stop] -= wgt[:C, None] * p_stop
        gT[:, self.pas] += wgt[C:, None] / self.pas.sum()
        gT = gT.reshape(np.shape(T))
        z = np.zeros_like(gT)
        return J, z, gT, z.copy()


class FunctionMerit:
    """Wrap a user function func(R, T, A) -> (J, dJ/dR, dJ/dT, dJ/dA);
    the arrays have the shape of the spectra, (conditions, L)."""

    def __init__(self, func):
        if not callable(func):
            raise ValueError("func must be callable")
        self.func = func

    def __call__(self, R, T, A):
        J, gR, gT, gA = self.func(R, T, A)
        return (float(J), np.asarray(gR, float), np.asarray(gT, float),
                np.asarray(gA, float))


@dataclasses.dataclass(frozen=True, eq=False)
class OpticalModel:
    """What is measured and how it is scored.

    merit : one of the merits above (called with spectra of shape
        (C, L), C = number of conditions).
    conditions : sequence of (theta0_rad, pol), pol "s", "p" or "u";
        default normal incidence.
    kext : None, (L,) or (N, L) extinction coefficients of the layers
        (fixed; the design varies thickness and the real index scale).
    n_exit : None (default: an infinitely thick substrate, as before)
        or the real index of the medium behind a thick, lossless
        substrate with a bare back face (1.0 for air; new in 0.8.0).
        The spectra are then those of the whole coated plate, as a
        spectrophotometer measures it (`fabtwin.tmm.stack_rt`).
    sub_thickness_um : None (default) or, with n_exit, the substrate
        thickness in um, which lets the substrate absorb (n_sub with
        k > 0; new in 0.12.0, see `fabtwin.tmm.stack_rt`).
    """

    merit: object
    conditions: tuple = ((0.0, "s"),)
    kext: object = None
    n_exit: object = None
    sub_thickness_um: object = None

    def __post_init__(self):
        c = self.conditions
        if isinstance(c, tuple) and len(c) == 2 and isinstance(c[1], str):
            c = (c,)                  # a single (angle, pol) pair
        try:
            conds = tuple((float(a), str(p)) for a, p in c)
        except (TypeError, ValueError):
            raise ValueError("conditions must be a sequence of "
                             "(angle_rad, pol) pairs") from None
        if not conds:
            raise ValueError("need at least one condition")
        for a, p in conds:
            if not (0.0 <= a < np.pi / 2) or p not in ("s", "p", "u"):
                raise ValueError("each condition is (angle in [0, pi/2), "
                                 "'s' | 'p' | 'u')")
        object.__setattr__(self, "conditions", conds)
        if not callable(self.merit):
            raise ValueError("merit must be one of the fabtwin merits "
                             "(or callable like them)")
        if self.n_exit is not None:
            ne = np.asarray(self.n_exit)
            if np.iscomplexobj(ne) or ne.ndim > 1 or \
                    not np.all(np.isfinite(ne)) or np.any(ne <= 0):
                raise ValueError("n_exit must be None, or a positive real "
                                 "scalar or (L,) array")
        if self.sub_thickness_um is not None:
            if self.n_exit is None:
                raise ValueError("sub_thickness_um needs n_exit (it is the "
                                 "thickness of a thick substrate with a "
                                 "back face)")
            try:
                d = float(self.sub_thickness_um)
            except (TypeError, ValueError):
                d = float("nan")
            if not (np.isfinite(d) and d > 0):
                raise ValueError("sub_thickness_um must be positive and "
                                 "finite (um)")


def model_spectra(model, lam_um, t_um, n0, shape, n_inc=1.0, n_sub=1.0,
                  grads=False):
    """Spectra (C, L) of R, T, A under every condition (and, with
    grads=True, the per-condition gradient dicts)."""
    n = layer_indices(n0, shape, model.kext)
    out = [stack_rta_and_grads(lam_um, t_um, n, n_inc, n_sub, a, p,
                               n_exit=model.n_exit,
                               sub_thickness_um=model.sub_thickness_um)
           for a, p in model.conditions]
    R = np.array([o["R"] for o in out])
    T = np.array([o["T"] for o in out])
    A = np.array([o["A"] for o in out])
    return (R, T, A, out) if grads else (R, T, A)


# Many designs at once (new in 0.13.0). Up to 0.12.0 every scoring loop
# with model= (evaluate_under_process, induced_merits, random_search, the
# probes of inverse_design, and every draw of robustify) called
# model_spectra or model_merit_and_grad once per design. Here the designs
# sit side by side along the wavelength axis of one solver call (column
# k * L + l is design k at wavelength l), in chunks of about
# _BATCH_COLUMNS layer-column values, as `fabtwin.tmm._transmittance_batch`
# does for the linear merit. Every column goes through the same
# element-wise arithmetic as a single-design call; only the merit itself
# is evaluated one design at a time. The tests compare the two to 1e-12 of
# the largest value.
_BATCH_COLUMNS = 2 ** 15


def _tile_medium(v, k):
    """A scalar medium index as it is, an (L,) array repeated k times."""
    a = np.asarray(v)
    return v if a.ndim == 0 else np.tile(a, k)


def _model_merits_batch(model, lam_um, t_k, n_k, shape, n_inc=1.0,
                        n_sub=1.0, grads=False):
    """The merit of K designs under an OpticalModel (new in 0.13.0).

    t_k, n_k : (K, N) thicknesses and indices at the reference
    wavelength; shape : (L,) or (N, L). Returns J (K,) and, with
    grads=True, also dJ/dt and dJ/dn0, each (K, N): equal, to rounding,
    to K calls of `model_merit_and_grad` (or of `model_spectra` and the
    merit)."""
    from .gradients import _rta_and_grads
    from .tmm import stack_rt
    lam = np.asarray(lam_um, dtype=float)
    t_k = np.asarray(t_k, dtype=float)
    n_k = np.asarray(n_k, dtype=float)
    if lam.ndim != 1 or t_k.ndim != 2 or n_k.shape != t_k.shape:
        raise ValueError("lam_um must be 1-D, and t_k and n_k (K, N) arrays "
                         "of one shape")
    K, N = t_k.shape
    L = lam.size
    S = np.asarray(shape)
    if np.iscomplexobj(S):
        raise ValueError("the dispersion shape must be real; give the "
                         "absorption through kext")
    S = S.astype(float)
    if S.ndim == 1:
        S = np.broadcast_to(S[None, :], (N, L))
    if S.shape != (N, L):
        raise ValueError("shape must be (L,) or (N, L)")
    # the extinction of the layers, checked as layer_indices checks it
    if np.iscomplexobj(np.asarray(n_inc)) and np.imag(n_inc) != 0:
        raise ValueError("the incidence medium must be lossless (real "
                         "index) for angle-resolved powers")
    if not (float(np.real(n_inc)) > 0):
        raise ValueError("n_inc must be positive")
    n_inc = float(np.real(n_inc))
    kx = None
    if model.kext is not None and K:
        kx = np.imag(layer_indices(n_k[0], S, model.kext))
    C = len(model.conditions)
    J = np.empty(K)
    gt = np.empty((K, N))
    gn = np.empty((K, N))
    step = max(1, int(_BATCH_COLUMNS // max(1, N * L)))
    for a in range(0, K, step):
        b = min(K, a + step)
        k = b - a
        lam_c = np.tile(lam, k)
        t_c = np.repeat(t_k[a:b].T, L, axis=1)                 # (N, k L)
        n_lay = n_k[a:b][:, :, None] * S[None, :, :]           # (k, N, L)
        if kx is not None:
            n_lay = n_lay + 1j * kx[None, :, :]
        n_c = n_lay.transpose(1, 0, 2).reshape(N, k * L)
        ns_c = _tile_medium(n_sub, k)
        ne_c = None if model.n_exit is None else \
            _tile_medium(model.n_exit, k)
        spectra = []
        for ang, pol in model.conditions:
            if grads:
                g = _rta_and_grads(lam_c, t_c, n_c, n_inc, ns_c, ang, pol,
                                   ne_c, model.sub_thickness_um)
            else:
                R_, T_ = stack_rt(lam_c, t_c, n_c, n_inc, ns_c, ang, pol,
                                  ne_c, model.sub_thickness_um)
                g = dict(R=R_, T=T_, A=1.0 - R_ - T_)
            spectra.append(g)
        R = np.stack([g["R"].reshape(k, L) for g in spectra], axis=1)
        T = np.stack([g["T"].reshape(k, L) for g in spectra], axis=1)
        A = np.stack([g["A"].reshape(k, L) for g in spectra], axis=1)
        if grads:
            d = {key: [g[key].reshape(N, k, L) for g in spectra]
                 for key in ("dR_dt", "dT_dt", "dA_dt", "dR_dn", "dT_dn",
                             "dA_dn")}
        for i in range(k):
            Ji, gR, gT, gA = model.merit(R[i], T[i], A[i])
            J[a + i] = float(Ji)
            if not grads:
                continue
            gR, gT, gA = (np.broadcast_to(gg, (C, L)) for gg in (gR, gT, gA))
            dt = np.zeros(N)
            dn = np.zeros(N)
            for c in range(C):
                dt += d["dR_dt"][c][:, i] @ gR[c] \
                    + d["dT_dt"][c][:, i] @ gT[c] \
                    + d["dA_dt"][c][:, i] @ gA[c]
                dn += (d["dR_dn"][c][:, i] * S) @ gR[c] \
                    + (d["dT_dn"][c][:, i] * S) @ gT[c] \
                    + (d["dA_dn"][c][:, i] * S) @ gA[c]
            gt[a + i] = dt
            gn[a + i] = dn
    return (J, gt, gn) if grads else J


def model_merit_and_grad(model, lam_um, t_um, n0, shape, n_inc=1.0,
                         n_sub=1.0):
    """J and its exact gradients (dJ/dt, dJ/dn0) for an OpticalModel."""
    S = np.asarray(shape, dtype=float)
    t = np.asarray(t_um, dtype=float)
    if S.ndim == 1:
        S = np.broadcast_to(S[None, :], (t.size, S.size))
    R, T, A, out = model_spectra(model, lam_um, t_um, n0, shape, n_inc,
                                 n_sub, grads=True)
    J, gR, gT, gA = model.merit(R, T, A)
    gR, gT, gA = (np.broadcast_to(g, R.shape) for g in (gR, gT, gA))
    dt = np.zeros(t.size)
    dn = np.zeros(t.size)
    for c, o in enumerate(out):
        dt += o["dR_dt"] @ gR[c] + o["dT_dt"] @ gT[c] + o["dA_dt"] @ gA[c]
        dn += (o["dR_dn"] * S) @ gR[c] + (o["dT_dn"] * S) @ gT[c] \
            + (o["dA_dn"] * S) @ gA[c]
    return float(J), dt, dn
