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
the deposition rate. The monitor beam arrives from the incidence medium
(`n_inc`, the chamber's vacuum, 1.0, by default) on the growing side
of the coating.
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np

from .gradients import _rta_and_grads

__all__ = ["MonitoringPlan", "monitoring_plan"]


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
