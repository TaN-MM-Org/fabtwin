"""Plan the optical monitoring of a deposition run (new in 0.10.0).

Many coating machines stop each layer with an optical monitor: a beam
at one wavelength passes through (or reflects from) the growing stack,
and the layer is stopped when the signal reaches a value computed in
advance from the recipe ("level monitoring"; see G. J. Kopec, "Thickness
of the layers of multilayer nonquarterwave interference filters
controlled by direct level monitoring", Optica Applicata 17(1), 33
(1987)). While a layer grows, the signal rises and falls; a maximum or
minimum is a "turning point". Near a turning point the signal hardly
changes with thickness, so a small error in the signal becomes a large
error in thickness. Which monitor wavelength to use is therefore a
planning question that can be answered from the recipe alone, before
any run.

`monitoring_plan` answers it for one monitor wavelength. For every
layer, in the order of deposition (layer N - 1, next to the substrate,
first), it computes the monitor signal along the growth of the layer
with the exact transfer-matrix optics (`fabtwin.tmm`), the cut-off
value at which to stop, the exact slope of the signal there
(`fabtwin.gradients`), the number of turning points passed during the
layer, and, for a stated noise of the signal, the thickness error that
noise causes at the cut-off:

    sigma_t = sigma_signal / |d signal / d t|,

the first-order (linear) propagation of the signal noise through the
inverse of the monitoring curve. This is valid only while the curve is
close to a straight line over +/- sigma_t; the second-order term is
|S''| sigma_t^2 / 2, so the ratio

    nonlinearity = |S''| sigma_signal / (2 S'^2)

measures how far the linear result can be trusted. Above
`max_nonlinearity` (0.1 by default) the layer ends too close to a
turning point for level monitoring at this wavelength: its `sigma_t`
is returned as NaN, with a RuntimeWarning, rather than as a number
that looks precise.

What the tests hold it to: the signal of one layer on a substrate
equals the closed form T(delta) = 4 n0 ns / [(n0 + ns)^2 cos^2 delta
+ (n0 ns / n1 + n1)^2 sin^2 delta] and its slope the derivative of
that closed form; its turning points sit at delta = m pi / 2 (counted
exactly); the cut-off values equal the transmittance of each partial
stack computed separately; a quarter-wave stack monitored at its
design wavelength ends every layer on a turning point (and is
refused); and the predicted sigma_t matches the scatter of simulated
stops, each found by solving signal(t) = cut-off - noise with the
forward optics alone.

Scope, stated plainly: the plan is for the recipe as designed. It
does not follow how an error made in one layer changes the curves of
the layers after it (in a real run the cut-offs are often recomputed
after every layer), and it does not model drift of the monitor or of
the deposition rate. Since 0.11.0, `monitoring_errors` carries the
errors of each layer into the stops of the layers after it (cut-offs
fixed from the recipe), and `simulate_monitored_runs` simulates such
runs without linearization; see their docstrings. The monitor beam arrives from the incidence medium
(`n_inc`, the chamber's vacuum, 1.0, by default) on the growing side
of the coating.
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np

from .gradients import _rta_and_grads

__all__ = ["MonitoringPlan", "monitoring_plan", "MonitoringErrors",
           "monitoring_errors", "simulate_monitored_runs"]


@dataclasses.dataclass
class MonitoringPlan:
    """The monitoring plan of one recipe at one monitor wavelength.

    Arrays of length N are indexed like the recipe (layer 0 next to the
    incidence medium); `order` gives the deposition order.

    lam_mon_um : the monitor wavelength (um); quantity : "T" or "R".
    order : (N,) layer indices in the order they are deposited
        (N - 1, ..., 0).
    start, cutoff : (N,) the signal when the layer starts and the
        value at which it is stopped (fractions, not percent).
    slope : (N,) d signal / d t at the cut-off (per um of the layer).
    turning_points : (N,) number of maxima and minima of the signal
        passed while the layer grows (not counting one that falls
        exactly at the start of the layer or on the cut-off).
    slope_ratio : (N,) |slope at the cut-off| / largest |slope| during
        the layer: 0 means the layer ends on a turning point, 1 at the
        steepest point of its curve.
    sigma_signal : the stated one-sigma noise of the signal (None if
        not given).
    sigma_t : (N,) predicted one-sigma thickness error at the cut-off
        (um), sigma_signal / |slope|; NaN where `nonlinearity` exceeds
        `max_nonlinearity` (or all NaN when sigma_signal is None).
    rel_sigma_t : (N,) sigma_t / t.
    nonlinearity : (N,) |S''| sigma_signal / (2 S'^2) at the cut-off.
    curves : list of N (t_grid (P,), signal (P,)) pairs, one per layer
        in recipe order, for plotting the monitoring curves.
    """

    lam_mon_um: float
    quantity: str
    order: np.ndarray
    start: np.ndarray
    cutoff: np.ndarray
    slope: np.ndarray
    turning_points: np.ndarray
    slope_ratio: np.ndarray
    sigma_signal: object
    sigma_t: np.ndarray
    rel_sigma_t: np.ndarray
    nonlinearity: np.ndarray
    curves: list

    def worst_rel_sigma_t(self) -> float:
        """The largest predicted relative thickness error (NaN if any
        layer cannot be monitored reliably at this wavelength)."""
        r = self.rel_sigma_t
        return float(np.nan) if np.any(np.isnan(r)) else float(np.max(r))


def _signals_and_slopes(lam_mon, taus, below_t, stack_n, n_inc, n_sub,
                        theta, pol, n_exit, quantity):
    """Signal and exact slope of the growing layer at every thickness in
    `taus` at once: the thicknesses sit along the wavelength axis of the
    gradient code (one column per thickness, all at lam_mon)."""
    P = taus.size
    lam = np.full(P, lam_mon)
    t2 = np.empty((1 + below_t.size, P))
    t2[0] = taus
    t2[1:] = below_t[:, None]
    n2 = np.repeat(np.asarray(stack_n).reshape(-1, 1), P, axis=1)
    g = _rta_and_grads(lam, t2, n2, n_inc, n_sub, theta, pol, n_exit)
    return g[quantity], g[f"d{quantity}_dt"][0]


def monitoring_plan(lam_mon_um: float, recipe_t_um, n_layers, n_inc=1.0,
                    n_sub=1.52, quantity: str = "T",
                    sigma_signal: float | None = None,
                    theta0_rad: float = 0.0, pol: str = "s", n_exit=None,
                    points_per_quarter_wave: int = 32,
                    max_nonlinearity: float = 0.1) -> MonitoringPlan:
    """Level-monitoring plan of a recipe at one monitor wavelength.

    lam_mon_um : the monitor wavelength (um).
    recipe_t_um : (N,) layer thicknesses (um), layer 0 next to the
        incidence medium; layer N - 1 (on the substrate) is deposited
        first.
    n_layers : (N,) indices of the layers AT THE MONITOR WAVELENGTH, real
        or n + i k (for a dispersive design: `n0 * S(lam_mon)`, see
        `fabtwin.layer_index`).
    n_inc : the medium the monitor beam arrives from (the chamber,
        1.0); n_sub : the substrate (or monitor glass) index at the
        monitor wavelength.
    quantity : "T" (transmission monitoring, the default) or "R".
    sigma_signal : one-sigma noise of the monitor signal, as a fraction
        (0.001 for 0.1 % in T). Optional; needed for sigma_t.
    theta0_rad, pol : angle and polarization of the monitor beam
        (as in `fabtwin.stack_rta_and_grads`).
    n_exit : None (an infinitely thick substrate) or the index behind a
        thick, non-absorbing substrate with a bare back face, for a
        monitor that measures through the whole plate (as in
        `fabtwin.stack_rt`).
    points_per_quarter_wave : sampling of each curve (at least 8), used
        to count turning points and to report the curves.
    max_nonlinearity : above this value of |S''| sigma_signal / (2 S'^2)
        the first-order sigma_t is not trustworthy and is returned as
        NaN with a RuntimeWarning.

    Returns a `MonitoringPlan`. Raises ValueError for a non-positive
    wavelength or thickness, mismatched shapes, a quantity other than
    "T" or "R", a non-positive sigma_signal, or too coarse a sampling.
    """
    lam_mon = float(lam_mon_um)
    if not (np.isfinite(lam_mon) and lam_mon > 0):
        raise ValueError("lam_mon_um must be a positive wavelength (um)")
    t = np.asarray(recipe_t_um, dtype=float)
    n = np.asarray(n_layers)
    if t.ndim != 1 or t.size < 1 or n.shape != t.shape:
        raise ValueError("recipe_t_um and n_layers must be (N,) arrays of "
                         "one length (indices at the monitor wavelength)")
    if not np.all(np.isfinite(t)) or np.any(t <= 0):
        raise ValueError("recipe thicknesses must be positive and finite")
    if not np.all(np.isfinite(n)) or np.any(np.real(n) <= 0):
        raise ValueError("layer indices must be finite with a positive "
                         "real part")
    if quantity not in ("T", "R"):
        raise ValueError("quantity must be 'T' or 'R'")
    if sigma_signal is not None:
        sigma_signal = float(sigma_signal)
        if not (np.isfinite(sigma_signal) and sigma_signal > 0):
            raise ValueError("sigma_signal must be positive and finite "
                             "(a fraction: 0.001 for 0.1 %)")
    ppq = int(points_per_quarter_wave)
    if ppq < 8:
        raise ValueError("points_per_quarter_wave must be at least 8")
    mnl = float(max_nonlinearity)
    if not (mnl > 0):
        raise ValueError("max_nonlinearity must be positive")
    N = t.size
    s0 = float(np.real(n_inc)) * np.sin(float(theta0_rad))
    order = np.arange(N - 1, -1, -1)
    start = np.empty(N)
    cutoff = np.empty(N)
    slope = np.empty(N)
    tp = np.zeros(N, dtype=int)
    ratio = np.empty(N)
    nonlin = np.empty(N)
    curves = [None] * N
    for j in order:
        below_t = t[j + 1:]
        # quarter-wave thickness of the growing layer at this angle
        q = np.sqrt(complex(n[j]) ** 2 - s0 ** 2)
        qw = lam_mon / (4.0 * max(abs(np.real(q)), 1e-12))
        P = max(2 * ppq, int(np.ceil(ppq * t[j] / qw)) + 1)
        grid = np.linspace(0.0, t[j], P + 1)
        h = 1e-4 * qw
        lo = max(t[j] - h, 0.0)
        vals, slopes = _signals_and_slopes(
            lam_mon, np.concatenate([grid, [lo, t[j] + h]]), below_t, n[j:],
            n_inc, n_sub, theta0_rad, pol, n_exit, quantity)
        # S'' at the cut-off by a central difference of the exact slope
        S2 = (slopes[-1] - slopes[-2]) / (t[j] + h - lo)
        vals, slopes = vals[:-2], slopes[:-2]
        start[j], cutoff[j], slope[j] = vals[0], vals[-1], slopes[-1]
        smax = float(np.max(np.abs(slopes)))
        ratio[j] = abs(slope[j]) / smax if smax > 0 else 0.0
        # turning points: sign changes of the exact slope inside the
        # layer. A slope that is zero to round-off (below 1e-9 of the
        # largest) carries no sign, so a turning point exactly at the
        # start or at the cut-off is not counted, and one exactly on a
        # grid point inside is counted once
        sgn = np.where(np.abs(slopes) <= 1e-9 * smax, 0.0, np.sign(slopes))
        sgn = sgn[sgn != 0]
        tp[j] = int(np.sum(sgn[1:] != sgn[:-1]))
        if sigma_signal is None:
            nonlin[j] = np.nan
        elif slope[j] == 0.0:
            nonlin[j] = np.inf
        else:
            nonlin[j] = abs(S2) * sigma_signal / (2.0 * slope[j] ** 2)
        curves[j] = (grid, vals)
    if sigma_signal is None:
        sigma_t = np.full(N, np.nan)
    else:
        with np.errstate(divide="ignore"):
            sigma_t = sigma_signal / np.abs(slope)
        bad = ~(nonlin <= mnl)
        if np.any(bad):
            sigma_t = np.where(bad, np.nan, sigma_t)
            names = ", ".join(str(int(i)) for i in np.nonzero(bad)[0])
            warnings.warn(
                f"layer(s) {names} end too close to a turning point of the "
                f"monitor signal at {lam_mon:g} um (nonlinearity above "
                f"{mnl:g}): a first-order thickness error would not be "
                "trustworthy, so it is returned as NaN. Monitor at another "
                "wavelength, or stop these layers at the turning point "
                "(turning-point monitoring)", RuntimeWarning, stacklevel=2)
    return MonitoringPlan(
        lam_mon_um=lam_mon, quantity=quantity, order=order, start=start,
        cutoff=cutoff, slope=slope, turning_points=tp, slope_ratio=ratio,
        sigma_signal=sigma_signal, sigma_t=sigma_t, rel_sigma_t=sigma_t / t,
        nonlinearity=nonlin, curves=curves)


# ----------------------------------------------------------------------
# Whole runs: how the errors of one layer carry into the next (0.11.0)


@dataclasses.dataclass
class MonitoringErrors:
    """The thickness errors of a whole level-monitored run, to first
    order in the signal errors (from `monitoring_errors`).

    Arrays of length N are indexed like the recipe (layer 0 next to the
    incidence medium; layer N - 1 is deposited first, see `order`);
    thicknesses and their errors are in um.

    recipe_t_um : (N,) the recipe; lam_mon_um, quantity : as in
        `MonitoringPlan`; order : (N,) deposition order.
    sigma_signal, signal_offset : the stated one-sigma noise of the
        monitor signal at each stop, and a constant offset of the
        measured signal (a calibration error), both as fractions.
    sensitivity : (N, N) A[j, k] = d S_j / d t_k at the recipe, where
        S_j is the true signal at the moment layer j is stopped; it
        depends on layer j and on the layers below it (k >= j), so A is
        upper triangular.
    propagation : (N, N) G = -A^-1, the thickness errors per unit
        signal error at each stop: dt = G (e + signal_offset), with e
        the (N,) noise at the stops.
    mean_t : (N,) G applied to the offset (zero without an offset).
    covariance : (N, N) sigma_signal^2 G G^T; sigma_t : (N,) the square
        roots of its diagonal; correlation : (N, N).
    rel_mean_t, rel_sigma_t, rel_covariance : the same for the relative
        errors dt / t (the package's error vectors).
    sigma_t_independent : (N,) sigma_signal / |A[j, j]|, the error each
        layer would have if the layers below it were exact; this is the
        `sigma_t` of `monitoring_plan`.
    nonlinearity : (N,) |S_j''| sqrt(sigma_t^2 + mean_t^2) / (2 |A[j, j]|),
        the size of the second-order term of layer j's own monitoring
        curve over its error, relative to the first-order term.
    """

    recipe_t_um: np.ndarray
    lam_mon_um: float
    quantity: str
    order: np.ndarray
    sigma_signal: float
    signal_offset: float
    sensitivity: np.ndarray
    propagation: np.ndarray
    mean_t: np.ndarray
    covariance: np.ndarray
    sigma_t: np.ndarray
    correlation: np.ndarray
    rel_mean_t: np.ndarray
    rel_sigma_t: np.ndarray
    rel_covariance: np.ndarray
    sigma_t_independent: np.ndarray
    nonlinearity: np.ndarray

    def twin(self):
        """A `GaussianTwin` of this monitored run: relative thickness
        errors with mean `rel_mean_t` and covariance `rel_covariance`,
        index errors exactly zero (the monitor model says nothing about
        them). It describes runs of THIS recipe; a design that moves far
        from it has other monitoring curves and needs its own."""
        from .twins import GaussianTwin
        N = self.recipe_t_um.size
        tw = GaussianTwin.__new__(GaussianTwin)
        tw.dim = 2 * N
        tw.mu = np.concatenate([self.rel_mean_t, np.zeros(N)])
        cov = np.zeros((2 * N, 2 * N))
        C = 0.5 * (self.rel_covariance + self.rel_covariance.T)
        cov[:N, :N] = C
        tw.cov = cov
        L = np.zeros((2 * N, 2 * N))
        L[:N, :N] = np.linalg.cholesky(C)
        tw.L = L
        tw.diagonal = False
        return tw


def _check_run_inputs(lam_mon_um, recipe_t_um, n_layers, quantity):
    """The input checks of `monitoring_plan`, for the whole-run tools."""
    lam_mon = float(lam_mon_um)
    if not (np.isfinite(lam_mon) and lam_mon > 0):
        raise ValueError("lam_mon_um must be a positive wavelength (um)")
    t = np.asarray(recipe_t_um, dtype=float)
    n = np.asarray(n_layers)
    if t.ndim != 1 or t.size < 1 or n.shape != t.shape:
        raise ValueError("recipe_t_um and n_layers must be (N,) arrays of "
                         "one length (indices at the monitor wavelength)")
    if not np.all(np.isfinite(t)) or np.any(t <= 0):
        raise ValueError("recipe thicknesses must be positive and finite")
    if not np.all(np.isfinite(n)) or np.any(np.real(n) <= 0):
        raise ValueError("layer indices must be finite with a positive "
                         "real part")
    if quantity not in ("T", "R"):
        raise ValueError("quantity must be 'T' or 'R'")
    return lam_mon, t, n


def _quarter_wave(lam_mon, nj, n_inc, theta0_rad):
    s0 = float(np.real(n_inc)) * np.sin(float(theta0_rad))
    q = np.sqrt(complex(nj) ** 2 - s0 ** 2)
    return lam_mon / (4.0 * max(abs(np.real(q)), 1e-12))


def _partial_signal(lam_mon, t_cols, n_part, n_inc, n_sub, theta, pol,
                    n_exit, quantity):
    """Signal of partial stacks, one per column of t_cols (Np, M) (top
    layer first), all of indices n_part (Np,), and the exact derivatives
    with respect to every layer thickness, (Np, M)."""
    M = t_cols.shape[1]
    lam = np.full(M, lam_mon)
    n2 = np.repeat(np.asarray(n_part).reshape(-1, 1), M, axis=1)
    g = _rta_and_grads(lam, t_cols, n2, n_inc, n_sub, theta, pol, n_exit)
    return g[quantity], g[f"d{quantity}_dt"]


def monitoring_errors(lam_mon_um: float, recipe_t_um, n_layers,
                      sigma_signal: float, n_inc=1.0, n_sub=1.52,
                      quantity: str = "T", theta0_rad: float = 0.0,
                      pol: str = "s", n_exit=None,
                      signal_offset: float = 0.0,
                      max_nonlinearity: float = 0.1) -> MonitoringErrors:
    """Thickness errors of a whole level-monitored run, with the errors
    of each layer carried into the layers after it (new in 0.11.0).

    `monitoring_plan` gives each layer's error as if the layers below it
    were exact. In a real run they are not: layer j is stopped when the
    signal of the stack AS DEPOSITED reaches the cut-off computed from
    the recipe, so the errors already made shift its stop. To first
    order, with S_j the true signal when layer j is stopped (a function
    of t_j and of the thicknesses below it), e_j the signal noise at
    that stop and o a constant offset of the measured signal,

        sum over k >= j of A[j, k] dt_k = -(e_j + o),
        A[j, k] = d S_j / d t_k at the recipe,

    so dt = G (e + o) with G = -A^-1, and for independent noise of
    standard deviation sigma_signal at every stop

        mean(dt) = o G 1,   cov(dt) = sigma_signal^2 G G^T.

    The derivatives are the exact ones of `fabtwin.gradients`. The
    errors of different layers are correlated, and an error can grow or
    partly cancel from layer to layer, depending on the monitor
    wavelength.

    lam_mon_um, recipe_t_um, n_layers, n_inc, n_sub, quantity,
    theta0_rad, pol, n_exit : as in `monitoring_plan` (indices at the
        monitor wavelength).
    sigma_signal : one-sigma noise of the signal at each stop, as a
        fraction (required, > 0).
    signal_offset : a constant error of the measured signal, the same at
        every stop (for example a calibration offset of the monitor), as
        a fraction; default 0.
    max_nonlinearity : refusal threshold on `nonlinearity` (as in
        `monitoring_plan`, but with the propagated error of each layer).

    Returns a `MonitoringErrors`; `.twin()` turns it into a
    `GaussianTwin` for `evaluate_under_process` or `robustify`.

    Refused (ValueError): the input errors `monitoring_plan` refuses, a
    sigma_signal that is not positive, a non-finite offset, a layer whose
    signal does not change at its cut-off (A[j, j] = 0), and a run in
    which some layer ends so close to a turning point that the first
    order cannot be trusted (`nonlinearity` above `max_nonlinearity`;
    the message names the layers). The model: the cut-offs are fixed
    from the recipe before the run (not recomputed during it), the
    noise at different stops is independent, and nothing else
    (deposition rate, index, monitor drift) changes during the run.
    `simulate_monitored_runs` simulates the same model without the
    linearization; the tests compare the two.
    """
    lam_mon, t, n = _check_run_inputs(lam_mon_um, recipe_t_um, n_layers,
                                      quantity)
    sig = float(sigma_signal)
    if not (np.isfinite(sig) and sig > 0):
        raise ValueError("sigma_signal must be positive and finite (a "
                         "fraction: 0.001 for 0.1 %)")
    off = float(signal_offset)
    if not np.isfinite(off):
        raise ValueError("signal_offset must be finite")
    mnl = float(max_nonlinearity)
    if not (mnl > 0):
        raise ValueError("max_nonlinearity must be positive")
    N = t.size
    A = np.zeros((N, N))
    S2 = np.empty(N)
    for j in range(N):
        _, d = _partial_signal(lam_mon, t[j:, None], n[j:], n_inc, n_sub,
                               theta0_rad, pol, n_exit, quantity)
        A[j, j:] = d[:, 0]
        # S'' of layer j's own curve: a central difference of the exact
        # slope, as in monitoring_plan
        h = 1e-4 * _quarter_wave(lam_mon, n[j], n_inc, theta0_rad)
        lo = max(t[j] - h, 0.0)
        cols = np.repeat(t[j:, None], 2, axis=1)
        cols[0] = (lo, t[j] + h)
        _, d2 = _partial_signal(lam_mon, cols, n[j:], n_inc, n_sub,
                                theta0_rad, pol, n_exit, quantity)
        S2[j] = (d2[0, 1] - d2[0, 0]) / (t[j] + h - lo)
    diag = np.diag(A)
    if np.any(diag == 0.0):
        bad = ", ".join(str(i) for i in np.nonzero(diag == 0.0)[0])
        raise ValueError(f"layer(s) {bad}: the monitor signal does not "
                         "change at the cut-off (a turning point), so the "
                         "stop is not determined to first order")
    # A is upper triangular: G = -A^-1 by back substitution
    from scipy.linalg import solve_triangular
    G = -solve_triangular(A, np.eye(N), lower=False)
    mean = G @ np.full(N, off)
    cov = sig ** 2 * (G @ G.T)
    cov = 0.5 * (cov + cov.T)
    sd = np.sqrt(np.diag(cov))
    nonlin = np.abs(S2) * np.sqrt(sd ** 2 + mean ** 2) / (2.0 * np.abs(diag))
    bad = ~(nonlin <= mnl)
    if np.any(bad):
        names = ", ".join(str(int(i)) for i in np.nonzero(bad)[0])
        raise ValueError(
            f"layer(s) {names} end too close to a turning point of the "
            f"monitor signal at {lam_mon:g} um for the errors they receive "
            f"(nonlinearity above {mnl:g}): a first-order result would not "
            "be trustworthy. Monitor at another wavelength, or use "
            "simulate_monitored_runs")
    rel_cov = cov / np.outer(t, t)
    return MonitoringErrors(
        recipe_t_um=t.copy(), lam_mon_um=lam_mon, quantity=quantity,
        order=np.arange(N - 1, -1, -1), sigma_signal=sig,
        signal_offset=off, sensitivity=A, propagation=G, mean_t=mean,
        covariance=cov, sigma_t=sd, correlation=cov / np.outer(sd, sd),
        rel_mean_t=mean / t, rel_sigma_t=sd / t, rel_covariance=rel_cov,
        sigma_t_independent=sig / np.abs(diag), nonlinearity=nonlin)


def simulate_monitored_runs(lam_mon_um: float, recipe_t_um, n_layers,
                            sigma_signal: float | None = None,
                            n_runs: int | None = None, seed=0, noise=None,
                            n_inc=1.0, n_sub=1.52, quantity: str = "T",
                            theta0_rad: float = 0.0, pol: str = "s",
                            n_exit=None,
                            signal_offset: float = 0.0) -> np.ndarray:
    """Simulate level-monitored runs layer by layer, without any
    linearization (new in 0.11.0).

    Each run deposits the layers in order (layer N - 1 first). Layer j
    grows on the stack as actually deposited in that run and is stopped
    when the measured signal, true signal + signal_offset + noise,
    reaches the cut-off computed from the recipe (the value
    `monitoring_plan` reports): the stop thickness solves

        S_j(tau; layers below as deposited) = cutoff_j - offset - e_j

    on the branch of layer j's monitoring curve that contains the recipe
    thickness (between its neighboring turning points), by bisection on
    the forward optics (`fabtwin.tmm.stack_rt`, one column per run), to
    about 1e-15 um; the exact slopes of `fabtwin.gradients` only locate
    the turning points.

    lam_mon_um, recipe_t_um, n_layers, n_inc, n_sub, quantity,
    theta0_rad, pol, n_exit, signal_offset : as in `monitoring_errors`.
    sigma_signal, n_runs, seed : draw the noise e as independent normal
        values of standard deviation sigma_signal, (n_runs, N), from
        `numpy.random.default_rng(seed)` (seed may also be a Generator);
    noise : or give the noise at every stop directly, (n_runs, N), with
        noise[r, j] the signal error when layer j of run r is stopped.

    Returns (n_runs, N) deposited thicknesses (um), indexed like the
    recipe. A stop that cannot happen on that branch (the noise or the
    errors below move the target past a turning point, so the signal
    turns before reaching it, or had passed it before the branch began)
    makes that layer and the layers after it in that run NaN, with a
    RuntimeWarning that counts such runs. A real controller that counts
    turning points would stop such a layer at the turning point or miss
    it; either way the run is not the level-monitored run modeled here.
    """
    lam_mon, t, n = _check_run_inputs(lam_mon_um, recipe_t_um, n_layers,
                                      quantity)
    N = t.size
    off = float(signal_offset)
    if not np.isfinite(off):
        raise ValueError("signal_offset must be finite")
    if noise is not None:
        if sigma_signal is not None or n_runs is not None:
            raise ValueError("give either noise, or sigma_signal and "
                             "n_runs, not both")
        e = np.asarray(noise, dtype=float)
        if e.ndim != 2 or e.shape[1] != N or e.shape[0] < 1 or \
                not np.all(np.isfinite(e)):
            raise ValueError("noise must be a finite (n_runs, N) array")
    else:
        if sigma_signal is None or n_runs is None:
            raise ValueError("give sigma_signal and n_runs (or noise)")
        sig = float(sigma_signal)
        if not (np.isfinite(sig) and sig >= 0):
            raise ValueError("sigma_signal must be finite and >= 0")
        if int(n_runs) != n_runs or int(n_runs) < 1:
            raise ValueError("n_runs must be a whole number >= 1")
        rng = seed if isinstance(seed, np.random.Generator) else \
            np.random.default_rng(seed)
        e = rng.normal(0.0, 1.0, (int(n_runs), N)) * sig
    K = e.shape[0]
    out = np.full((K, N), np.nan)
    alive = np.ones(K, dtype=bool)
    eps = np.finfo(float).eps

    from .tmm import stack_rt
    qi = 0 if quantity == "R" else 1

    def columns(j, tau, rows):
        cols = np.empty((N - j, rows.size))
        cols[0] = tau
        cols[1:] = out[rows][:, j + 1:].T
        return cols

    def val_at(j, tau, rows):
        # the signal from the forward solver alone (stack_rt, one column
        # per run); the gradient module is used only for slope signs
        M = rows.size
        n2 = np.repeat(np.asarray(n[j:]).reshape(-1, 1), M, axis=1)
        return stack_rt(np.full(M, lam_mon), columns(j, tau, rows), n2,
                        n_inc, n_sub, theta0_rad, pol, n_exit)[qi]

    def sig_at(j, tau, rows):
        _, d = _partial_signal(lam_mon, columns(j, tau, rows), n[j:], n_inc,
                               n_sub, theta0_rad, pol, n_exit, quantity)
        return val_at(j, tau, rows), d[0]

    for j in range(N - 1, -1, -1):
        cut = stack_rt([lam_mon], t[j:], n[j:], n_inc, n_sub, theta0_rad,
                       pol, n_exit)[qi]
        rows = np.nonzero(alive)[0]
        if rows.size == 0:
            break
        target = cut[0] - off - e[rows, j]
        qw = _quarter_wave(lam_mon, n[j], n_inc, theta0_rad)
        x0 = np.full(rows.size, t[j])
        v0, s0 = sig_at(j, x0, rows)
        f0 = v0 - target
        sgn_s = np.sign(s0)
        # direction from the recipe thickness towards the root on this
        # branch of the curve
        d = np.where(f0 * s0 > 0, -1.0, 1.0)
        a = x0.copy()
        b = x0.copy()
        done = (f0 == 0.0) | (s0 == 0.0)
        missed = s0 == 0.0
        step = np.abs(f0) / np.where(s0 != 0, np.abs(s0), 1.0) * 1.5 \
            + 1e-9 * qw
        found = done.copy()
        for _ in range(80):
            todo = ~found
            if not np.any(todo):
                break
            nb = np.maximum(a[todo] + d[todo] * step[todo], 0.0)
            vb, sb = sig_at(j, nb, rows[todo])
            fnb = vb - target[todo]
            turned = np.sign(sb) != sgn_s[todo]
            crossed = (np.sign(fnb) != np.sign(f0[todo])) & ~turned
            idx = np.nonzero(todo)[0]
            # a turning point between a and nb: find it (bisection on
            # the sign of the slope); the root is before it only if the
            # signal crosses the target before turning
            if np.any(turned):
                ti = idx[turned]
                lo_, hi_ = a[ti].copy(), nb[turned].copy()
                for _ in range(200):
                    mid = 0.5 * (lo_ + hi_)
                    _, sm = sig_at(j, mid, rows[ti])
                    same = np.sign(sm) == sgn_s[ti]
                    lo_ = np.where(same, mid, lo_)
                    hi_ = np.where(same, hi_, mid)
                    if np.all(np.abs(hi_ - lo_) <= 4 * eps * hi_):
                        break
                vt = val_at(j, lo_, rows[ti])
                ok = np.sign(vt - target[ti]) != np.sign(f0[ti])
                b[ti[ok]] = lo_[ok]
                found[ti] = True
                missed[ti[~ok]] = True
            at_zero = (nb == 0.0) & ~crossed & ~turned
            if np.any(at_zero):
                missed[idx[at_zero]] = True
                found[idx[at_zero]] = True
            ci = idx[crossed]
            b[ci] = nb[crossed]
            found[ci] = True
            keep = idx[~crossed & ~turned & ~at_zero]
            a[keep] = nb[~crossed & ~turned & ~at_zero]
            step[keep] *= 2.0
            far = np.abs(a[keep] - t[j]) > 2.0 * qw
            missed[keep[far]] = True
            found[keep[far]] = True
        missed |= ~found
        # bisection between a and b, with f(a) on the side of f0
        lo_ = np.minimum(a, b)
        hi_ = np.maximum(a, b)
        live = ~done & ~missed
        for _ in range(200):
            if not np.any(live):
                break
            li = np.nonzero(live)[0]
            mid = 0.5 * (lo_[li] + hi_[li])
            vm = val_at(j, mid, rows[li])
            fm = vm - target[li]
            # f is monotone on [lo, hi], increasing when the slope sign
            # is +1: the root is right of mid when f(mid) has the sign
            # of -slope
            left = np.sign(fm) == np.sign(-sgn_s[li])
            lo_[li] = np.where(left, mid, lo_[li])
            hi_[li] = np.where(left, hi_[li], mid)
            live[li] = np.abs(hi_[li] - lo_[li]) > 4 * eps * hi_[li]
        stop = np.where(done & ~missed, x0, 0.5 * (lo_ + hi_))
        good = ~missed
        out[rows[good], j] = stop[good]
        alive[rows[missed]] = False
    n_missed = int(np.sum(~alive))
    if n_missed:
        warnings.warn(
            f"{n_missed} of {K} simulated runs missed a stop (the target "
            "signal is not reached on the branch of that layer's curve "
            "between its turning points); their remaining thicknesses "
            "are NaN", RuntimeWarning,
            stacklevel=2)
    return out
