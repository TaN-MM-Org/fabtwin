"""Exact transfer-matrix optics for arbitrary stratified stacks.

Characteristic-matrix formulation (H. A. Macleod, Thin-Film Optical
Filters, 4th ed., 2010): for layers i = 1..N between an incidence
medium and a substrate,

    [B; C] = { prod_i [[cos d_i, i sin d_i / eta_i],
                       [i eta_i sin d_i, cos d_i]] } [1; eta_sub],

with phase thickness d_i = 2 pi n_i t_i cos(theta_i) / lam and tilted
admittances eta = n cos(theta) (s) or n / cos(theta) (p); Snell's law
n_inc sin(theta_0) = n_i sin(theta_i) continued into the complex plane
for absorbing layers. Reflectance and transmittance follow

    r = (eta_0 B - C) / (eta_0 B + C),   R = |r|^2,
    t = 2 eta_0 / (eta_0 B + C),         T = 4 eta_0 Re(eta_sub) / |eta_0 B + C|^2.

Everything here is closed-form linear algebra -- no fitting, no
surrogates -- and the tests hold it to closed forms rather than trust
it: the bare-interface Fresnel transmittance, the exact absentee
half-wave layer, the quarter-wave admittance transform, R + T = 1 for
lossless stacks at machine precision, the analytic Brewster zero of
p-polarized interface reflectance, the equality of s and p at normal
incidence, and agreement with the independent open-source `tmm`
reference (S. J. Byrnes, arXiv:1603.02720) on random stacks.

A thick substrate with a bare back face (new in 0.8.0, `n_exit=`):
the coherent coating above is combined with the substrate's back face
by summing the multiple reflections between them in power
(incoherently), which is how a spectrophotometer sees a coated plate
much thicker than the coherence length of its light; the tests hold
this to the incoherent solver `inc_tmm` of the same `tmm` reference
and to the closed-form bare plate T = 2n / (n^2 + 1). Since 0.12.0 the
substrate may absorb when its thickness is given (`sub_thickness_um`):
each pass through it keeps exp(-4 pi d Im(n cos theta) / lam) of the
power, held in the tests to `inc_tmm` with an absorbing incoherent
layer and to the closed form of a bare absorbing plate.

The merit functions here are *linear in T* (weights plus a
constant), so the exact adjoint of `fabtwin.adjoint` needs only the
weight vector (nonlinear merits of R, T and A at any angle live in
`fabtwin.merits`, new in 0.7.0); `notch_weights` and `bandpass_weights` build the two
sensor-front-end merits of the FabGAN-ID study (Mahim et al., IEEE
Sensors J., 2026) for any band layout, and any user-supplied weight
vector works identically.
"""
from __future__ import annotations

import numpy as np

__all__ = ["stack_BC", "stack_rt", "transmittance", "reflectance",
           "notch_weights", "bandpass_weights", "merit",
           "weights_from_reflectance"]


def _cos_branch(n, s0):
    """cos(theta) in a medium of (conjugated) index n, with n cos(theta)
    on the physical (fourth-quadrant) branch."""
    cos = np.sqrt(1.0 - (s0 / n) ** 2 + 0.0j)      # principal root
    ncos = n * cos
    flip = (np.imag(ncos) > 0.0) & (np.abs(np.real(ncos))
                                    <= 1e-14 * np.abs(ncos))
    return np.where(flip, -cos, cos)


def _phases_and_admittances(lam_um, t_um, n_layers, n_inc, n_sub,
                            theta0_rad, pol, s0=None):
    # s0 (internal, new in 0.8.0): the Snell invariant n sin(theta),
    # given directly when the incidence medium is a dispersive (L,)
    # array -- the substrate seen from inside, for the thick-substrate
    # calculation. The public path (s0=None) is unchanged.
    lam = np.asarray(lam_um, dtype=float)
    t = np.asarray(t_um, dtype=float)
    n = np.asarray(n_layers)
    if n.ndim == 1:                     # dispersionless layers
        n = np.repeat(n[:, None], lam.size, axis=1)
    if t.ndim not in (1, 2) or (t.ndim == 2 and t.shape[1] != lam.size):
        raise ValueError("t_um must be (N,) layer thicknesses")
    if n.shape != (t.shape[0], lam.size):
        raise ValueError("n_layers must be (N,) or (N, L) matching "
                         "t_um and lam_um")
    # convention: n + i k with k >= 0 is ABSORBING (the common Python
    # ecosystem convention, e.g. the open tmm package); the Macleod
    # characteristic-matrix recursion below wants n - i k, so conjugate
    # internally. Gain media (k < 0) are refused, not extrapolated.
    if s0 is None:
        inc_gain = np.imag(complex(n_inc)) < 0
    else:
        inc_gain = np.any(np.imag(np.asarray(n_inc)) < 0)
    if np.any(np.imag(n) < 0) or inc_gain \
            or np.any(np.imag(np.asarray(n_sub)) < 0):
        raise ValueError("negative Im(n) (gain) is out of scope; "
                         "absorbing media carry n + i k with k >= 0")
    if s0 is None and np.imag(complex(n_inc)) != 0 \
            and float(theta0_rad) != 0.0:
        # (new in 0.12.0) in an absorbing incidence medium n sin(theta)
        # is complex: the incident wave is not a plain plane wave at a
        # real angle, and the independent `tmm` package refuses this
        # case too. 0.11.0 returned numbers for it.
        raise ValueError(
            "oblique incidence from an absorbing incidence medium (n_inc "
            "with k > 0) is out of scope: n_inc sin(theta0) would be "
            "complex. Use a lossless incidence medium, or normal incidence")
    n = np.conj(n)
    if s0 is None:
        n_inc = np.conj(complex(n_inc))
    else:
        n_inc = np.conj(np.asarray(n_inc)) + 0.0j
    n_sub_arr = np.conj(np.asarray(n_sub)) + 0.0j
    if n_sub_arr.ndim == 0:
        n_sub_arr = np.full(lam.size, n_sub_arr)
    if pol not in ("s", "p"):
        raise ValueError("pol must be 's' or 'p' ('u' in stack_rt, "
                         "transmittance and reflectance)")
    if s0 is None:
        s0 = n_inc * np.sin(float(theta0_rad))
    # complex Snell cosines. In the conjugated (n - i k) convention a
    # wave travelling or decaying away from the interface has
    # n cos(theta) in the fourth quadrant (Re >= 0, Im <= 0). The
    # principal square root gives that automatically for absorbing
    # media, but for a LOSSLESS medium beyond the critical angle
    # (evanescent) it returns +i|.| on the branch cut; the physical
    # root is -i|.|. (0.6.1 and earlier took the principal root there,
    # which gave wrong R beyond the critical angle when the stack also
    # absorbed; see the 0.7.0 changelog.)
    cos_l = _cos_branch(n, s0)
    cos_sub = _cos_branch(n_sub_arr, s0)
    cos_0 = _cos_branch(n_inc, s0)
    # t is (N,) on every public path; internally (new in 0.11.0) it may
    # be (N, L), one thickness per layer and column, which lets
    # _transmittance_batch put many designs side by side along the
    # column axis
    tc = t if t.ndim == 2 else t[:, None]
    delta = 2.0 * np.pi * n * cos_l * tc / lam[None, :]
    if pol == "p" and np.any(np.abs(cos_sub) == 0.0):
        raise ValueError("the substrate is exactly at its critical angle; "
                         "the p admittance n / cos(theta) is infinite "
                         "there")
    if pol == "s":
        eta = n * cos_l
        eta_sub = n_sub_arr * cos_sub
        eta0 = n_inc * cos_0
    else:
        eta = n / cos_l
        eta_sub = n_sub_arr / cos_sub
        eta0 = n_inc / cos_0
    return delta, eta, eta0, eta_sub


def stack_BC(lam_um, t_um, n_layers, n_inc=1.0, n_sub=1.0,
             theta0_rad=0.0, pol="s", _s0=None):
    """The Macleod (B, C) vectors and the admittances (eta0, eta_sub).

    lam_um : (L,) wavelengths; t_um : (N,) thicknesses;
    n_layers : (N,) or (N, L), real or complex.
    Returns B, C, eta0 (complex, possibly (L,)), eta_sub (L,).

    B and C grow like exp(sum |Im delta|) and overflow for an opaque
    stack (a thick metal or a thick evanescent layer); `stack_rt`
    rescales them and stays finite there (since 0.10.0), this function
    returns them unscaled, as before.
    """
    delta, eta, eta0, eta_sub = _phases_and_admittances(
        lam_um, t_um, n_layers, n_inc, n_sub, theta0_rad, pol, _s0)
    c = np.cos(delta)
    s = np.sin(delta)
    B = np.ones(delta.shape[1], dtype=complex)
    C = eta_sub.astype(complex).copy()
    # right-to-left accumulation of M_1 ... M_N [1; eta_sub]
    for i in range(delta.shape[0] - 1, -1, -1):
        Bi = c[i] * B + 1j * s[i] / eta[i] * C
        Ci = 1j * eta[i] * s[i] * B + c[i] * C
        B, C = Bi, Ci
    return B, C, eta0, eta_sub


# Opaque layers (new in 0.10.0). In a strongly absorbing or evanescent
# layer the phase delta has a large imaginary part m = |Im delta|, and
# cos(delta), sin(delta) grow like exp(m) / 2. Up to 0.9.x the product
# of the layer matrices overflowed: |B|, |C| ~ exp(sum m) gave a
# RuntimeWarning beyond sum m of about 355 (|eta0 B + C|^2 above the
# largest double) and NaN for R and T beyond about 709 (cos(delta)
# itself overflowing), for example 10 um of a metal with k = 6 at
# 0.5 um (m = 754). Every layer matrix may be multiplied by a constant
# without changing r = (eta0 B - C) / (eta0 B + C), and T changes by the
# square of the product of the constants, so in every wavelength column
# with sum m above _OPAQUE_SUM the layer matrices are scaled by
# exp(-m_i) (computed without overflow) and T is multiplied back by
# exp(-2 sum m). Columns at or below _OPAQUE_SUM take the unchanged
# path, bit for bit.
_OPAQUE_SUM = 300.0


def _scaled_cos_sin(delta):
    """cos(delta) exp(-|Im delta|) and sin(delta) exp(-|Im delta|),
    finite for any complex delta, and m = |Im delta|."""
    m = np.abs(np.imag(delta))
    ep = np.exp(1j * delta - m)          # both exponents have Re <= 0
    em = np.exp(-1j * delta - m)
    return 0.5 * (ep + em), (ep - em) / 2j, m


def _opaque_columns(delta):
    """(columns needing the scaled path, sum over layers of |Im delta|)."""
    msum = np.sum(np.abs(np.imag(delta)), axis=0)
    return msum > _OPAQUE_SUM, msum


def _incident_power_factor(eta0):
    """|eta0|^2 / Re(eta0)^2 where the incidence medium absorbs, else None
    (new in 0.12.0).

    The transmittance is the transmitted power divided by the power the
    incident wave carries towards the stack, Re(eta0) |E+|^2:

        T = |t|^2 Re(eta_sub) / Re(eta0),   t = 2 eta0 / (eta0 B + C),

    that is T = 4 |eta0|^2 Re(eta_sub) / (Re(eta0) |eta0 B + C|^2), the
    definition of the independent `tmm` package. For a lossless
    incidence medium eta0 is real and this is the familiar
    4 eta0 Re(eta_sub) / |eta0 B + C|^2. Up to 0.11.0 that real-eta0
    formula was used for an absorbing incidence medium too, which
    understates T by the factor Re(eta0)^2 / |eta0|^2."""
    e0 = np.asarray(eta0)
    if not (np.iscomplexobj(e0) and np.any(np.imag(e0) != 0)):
        return None
    return np.abs(e0) ** 2 / np.real(e0) ** 2


def _coherent_rt(lam_um, t_um, n_layers, n_inc, n_sub, theta0_rad, pol,
                 s0=None):
    delta, eta, eta0, eta_sub = _phases_and_admittances(
        lam_um, t_um, n_layers, n_inc, n_sub, theta0_rad, pol, s0)
    big, msum = _opaque_columns(delta)
    with np.errstate(over="ignore", invalid="ignore"):
        c = np.cos(delta)
        s = np.sin(delta)
    if np.any(big):
        cs, ss, _ = _scaled_cos_sin(delta[:, big])
        c[:, big] = cs
        s[:, big] = ss
    B = np.ones(delta.shape[1], dtype=complex)
    C = eta_sub.astype(complex).copy()
    for i in range(delta.shape[0] - 1, -1, -1):
        Bi = c[i] * B + 1j * s[i] / eta[i] * C
        Ci = 1j * eta[i] * s[i] * B + c[i] * C
        B, C = Bi, Ci
    denom = eta0 * B + C
    r = (eta0 * B - C) / denom
    R = np.abs(r) ** 2
    T = 4.0 * np.real(eta0) * np.real(eta_sub) / np.abs(denom) ** 2
    if np.any(big):
        T = np.where(big, T * np.exp(-2.0 * np.where(big, msum, 0.0)), T)
    fac = _incident_power_factor(eta0)
    if fac is not None:
        T = T * fac
    return R, T


# Many designs at once (new in 0.11.0). The per-design loops of
# `evaluate_under_process`, `induced_merits` and the probes of
# `inverse_design` and `random_search` called `transmittance` once per
# design. Here the designs sit side by side along the column axis of one
# call of the same solver (column k * L + l is design k at wavelength
# l), in chunks of about _BATCH_COLUMNS layer-column values. Every
# column goes through the same element-wise operations as the
# single-design call; the tests compare the two to 1e-12 of the largest
# value, not bit for bit, because vectorized math kernels may round the
# last digit differently on other processors.
_BATCH_COLUMNS = 2 ** 15


def _transmittance_batch(lam_um, t_k, n_k, shape, n_inc=1.0, n_sub=1.0):
    """Normal-incidence T (K, L) of K lossless or absorbing designs.

    t_k : (K, N) thicknesses; n_k : (K, N) indices at the reference
    wavelength; shape : (L,) or (N, L) real dispersion shape, so that
    design k has layer indices n_k[k][:, None] * shape, exactly as the
    per-design loops built them. Equal, to rounding, to K calls of
    `transmittance(lam_um, t_k[k], n_k[k][:, None] * shape, n_inc,
    n_sub)`.
    """
    lam = np.asarray(lam_um, dtype=float)
    t_k = np.asarray(t_k, dtype=float)
    n_k = np.asarray(n_k)
    if t_k.ndim != 2 or n_k.shape != t_k.shape:
        raise ValueError("t_k and n_k must be (K, N) arrays of one shape")
    K, N = t_k.shape
    L = lam.size
    S = np.asarray(shape)
    if S.ndim == 1:
        S = np.broadcast_to(S[None, :], (N, L))
    if S.shape != (N, L):
        raise ValueError("shape must be (L,) or (N, L)")
    ns = np.asarray(n_sub)
    if ns.ndim == 0:
        ns = np.full(L, ns)
    if ns.shape != (L,):
        raise ValueError("n_sub must be a scalar or (L,)")
    T = np.empty((K, L))
    step = max(1, int(_BATCH_COLUMNS // max(1, N * L)))
    for a in range(0, K, step):
        b = min(K, a + step)
        k = b - a
        tt = np.repeat(t_k[a:b].T, L, axis=1)                  # (N, k L)
        nn = (n_k[a:b].T[:, :, None] * S[:, None, :]).reshape(N, k * L)
        _, Tc = _coherent_rt(np.tile(lam, k), tt, nn, n_inc,
                             np.tile(ns, k), 0.0, "s")
        T[a:b] = Tc.reshape(k, L)
    return T


def _thick_substrate_media(lam_um, n_inc, n_sub, n_exit, theta0_rad,
                           sub_thickness_um=None):
    """Check and prepare the media of a thick-substrate calculation.

    Returns (n_inc float, n_sub (L,), n_exit (L,), s0, d) with
    s0 = n_inc sin(theta0), the Snell invariant, and d the substrate
    thickness (um, None when not given). Refuses an absorbing incidence
    or exit medium (the incoherent power sum below is written for
    lossless ones), an absorbing substrate without its thickness (since
    0.12.0 an absorbing substrate is allowed when `sub_thickness_um` is
    given: its single-pass loss then enters the power sum), and an
    angle at which no light travels in the substrate (s0 >= Re n_sub:
    nothing reaches the back face, so it plays no role; use
    n_exit=None)."""
    lam = np.asarray(lam_um, dtype=float)
    L = lam.size
    d = None
    if sub_thickness_um is not None:
        try:
            d = float(sub_thickness_um)
        except (TypeError, ValueError):
            raise ValueError("sub_thickness_um must be one positive number "
                             "(um)") from None
        if not (np.isfinite(d) and d > 0):
            raise ValueError("sub_thickness_um must be positive and finite "
                             "(um: 1000.0 for a 1 mm plate)")
    out = []
    for name, v in (("n_inc", n_inc), ("n_sub", n_sub),
                    ("n_exit", n_exit)):
        a = np.asarray(v)
        lossy = np.iscomplexobj(a) and np.any(np.imag(a) != 0)
        if lossy and name == "n_sub" and d is not None:
            if np.any(np.imag(a) < 0):
                raise ValueError("negative Im(n_sub) (gain) is out of "
                                 "scope; absorbing media carry n + i k "
                                 "with k >= 0")
            a = a.astype(complex)
        elif lossy:
            extra = (" (an absorbing substrate needs its thickness: give "
                     "sub_thickness_um)" if name == "n_sub" else "")
            raise ValueError(
                f"with n_exit (a thick substrate with a bare back face), "
                f"{name} must be lossless (real): the back-face "
                "reflections are summed in power for a non-absorbing "
                f"substrate{extra}")
        else:
            a = np.real(a).astype(float)
        if name == "n_inc":
            if a.ndim != 0:
                raise ValueError("n_inc must be a scalar with n_exit")
        elif a.ndim == 0:
            a = np.full(L, a[()])
        elif a.shape != (L,):
            raise ValueError(f"{name} must be a scalar or (L,)")
        if not np.all(np.isfinite(a)) or np.any(np.real(a) <= 0):
            raise ValueError(f"{name} must be positive and finite")
        out.append(a)
    ni, ns, ne = out
    s0 = float(ni) * np.sin(float(theta0_rad))
    if np.any(s0 >= np.real(ns)):
        raise ValueError(
            "no light travels in the substrate at this angle (beyond its "
            "critical angle), so its back face plays no role; use "
            "n_exit=None")
    return float(ni), ns, ne, s0, d


def _substrate_pass(lam, ns, s0, d):
    """Fraction of the power that survives one pass through a thick
    absorbing substrate (new in 0.12.0), or None for a lossless one.

        tau = exp(-4 pi d Im(q) / lam),   q = n cos(theta) = sqrt(n^2 - s0^2)

    on the decaying branch (Im q > 0 for n + i k), the power of a plane
    wave after the distance d along the normal. With d = None the
    substrate must be lossless (checked by _thick_substrate_media)."""
    if d is None or not (np.iscomplexobj(ns) and np.any(np.imag(ns) != 0)):
        return None
    nprime = np.conj(ns)                       # the solver's n - i k
    q = nprime * _cos_branch(nprime, s0)       # Im q <= 0 here
    return np.exp(4.0 * np.pi * d * np.imag(q) / lam)


def _thick_rt(lam_um, t_um, n_layers, n_inc, n_sub, n_exit, theta0_rad,
              pol, sub_thickness_um=None):
    lam = np.asarray(lam_um, dtype=float)
    ni, ns, ne, s0, d = _thick_substrate_media(lam, n_inc, n_sub, n_exit,
                                              theta0_rad, sub_thickness_um)
    t = np.asarray(t_um, dtype=float)
    n = np.asarray(n_layers)
    Rf, Tf = _coherent_rt(lam, t, n, ni, ns, theta0_rad, pol)
    # the same stack seen from inside the substrate (layers reversed)
    Rr, Tr = _coherent_rt(lam, t[::-1], n[::-1], ns, ni, 0.0, pol, s0)
    # the bare back face: substrate -> exit medium
    Rb, Tb = _coherent_rt(lam, np.zeros(0), np.zeros((0, lam.size)), ns,
                          ne, 0.0, pol, s0)
    tau = _substrate_pass(lam, ns, s0, d)
    if tau is not None:
        # one pass through the substrate on the way to the back face and
        # one on the way back (new in 0.12.0); lossless: skipped, so the
        # numbers are those of 0.11.0 bit for bit
        Rb, Tb = Rb * tau * tau, Tb * tau
        _check_round_trip(Rr * Rb)
    den = 1.0 - Rr * Rb
    return Rf + Tf * Tr * Rb / den, Tf * Tb / den


def _check_round_trip(g):
    """The power left after one round trip in an absorbing substrate is
    below 1 for any real plate; refuse otherwise rather than divide by
    a non-positive 1 - g."""
    if np.any(g >= 1.0):
        raise ValueError("the power sum of the substrate's multiple "
                         "reflections does not converge here (round-trip "
                         "factor >= 1); this case is out of scope")


def _check_angle(theta0_rad):
    """Refuse an angle of incidence that is not a finite number in
    [-pi/2, pi/2] (new in 0.9.0). Beyond pi/2 the light would arrive
    from behind; 0.8.0 silently returned the result for pi - theta."""
    try:
        th = float(theta0_rad)
    except (TypeError, ValueError):
        raise ValueError("theta0_rad must be one real number (radians)") \
            from None
    if not np.isfinite(th) or abs(th) > np.pi / 2:
        raise ValueError(
            f"theta0_rad = {theta0_rad!r} is not an angle of incidence: it "
            "must be finite and at most pi/2 in size (radians, not "
            "degrees; use np.deg2rad)")
    return th


def stack_rt(lam_um, t_um, n_layers, n_inc=1.0, n_sub=1.0,
             theta0_rad: float = 0.0, pol: str = "s",
             n_exit=None, sub_thickness_um=None) -> tuple:
    """Reflectance and transmittance (R, T) of the stack. pol is "s",
    "p" or (new in 0.7.0) "u", unpolarized light: the mean of s and p.

    n_exit (new in 0.8.0) : None (default) treats the substrate as
    infinitely thick, as before: R and T are the powers reflected by
    the coating and transmitted INTO the substrate. A number (or an
    (L,) array) instead makes the substrate a thick, lossless slab
    whose bare back face meets a medium of index n_exit (1.0 for air):
    R and T are then the powers leaving the whole sample, which is what
    a spectrophotometer measures on a coated glass plate. The light
    bouncing between the coating and the back face is added in power,
    not in amplitude (incoherently), as for a substrate much thicker
    than the coherence length of the measuring light:

        T = Tf Tb / (1 - Rr Rb),   R = Rf + Tf Tr Rb / (1 - Rr Rb),

    with Rf, Tf the coating seen from the incidence side, Rr, Tr the
    coating seen from inside the substrate, and Rb, Tb the back face.
    Unpolarized light is summed per polarization first. The
    incidence and exit media must be lossless.

    sub_thickness_um (new in 0.12.0) : with n_exit, the thickness of the
    substrate (um; 1000.0 for a 1 mm plate). It is needed, and only
    used, when the substrate absorbs (n_sub = n + i k with k > 0, for
    example a tinted or a slightly absorbing glass): each pass through
    it keeps the fraction tau = exp(-4 pi d Im(n cos theta) / lam) of
    the power, so Rb and Tb above become Rb tau^2 and Tb tau. Without
    it an absorbing substrate is refused, as before. For a lossless
    substrate it changes nothing.

    theta0_rad must be finite and at most pi/2 in size (new check in
    0.9.0).

    An absorbing incidence medium (n_inc = n + i k, k > 0) is allowed at
    normal incidence only (refused at an angle since 0.12.0). T is then
    the transmitted power divided by the power of the incident wave,
    Re(eta0) |E+|^2, as in the independent `tmm` package (0.11.0 and
    earlier used the lossless formula there, which understated T; see
    the 0.12.0 changelog). With an absorbing incidence medium, R + T is
    not 1 even for a lossless stack: in an absorbing medium the power
    flow of the incident and the reflected wave together is not the
    difference of their separate powers (they interfere).
    """
    _check_angle(theta0_rad)
    if sub_thickness_um is not None and n_exit is None:
        raise ValueError("sub_thickness_um is the thickness of a thick "
                         "substrate with a back face; it needs n_exit")
    if pol == "u":
        Rs, Ts = stack_rt(lam_um, t_um, n_layers, n_inc, n_sub,
                          theta0_rad, "s", n_exit, sub_thickness_um)
        Rp, Tp = stack_rt(lam_um, t_um, n_layers, n_inc, n_sub,
                          theta0_rad, "p", n_exit, sub_thickness_um)
        return 0.5 * (Rs + Rp), 0.5 * (Ts + Tp)
    if n_exit is not None:
        if pol not in ("s", "p"):
            raise ValueError("pol must be 's', 'p' or 'u'")
        return _thick_rt(lam_um, t_um, n_layers, n_inc, n_sub, n_exit,
                         theta0_rad, pol, sub_thickness_um)
    return _coherent_rt(lam_um, t_um, n_layers, n_inc, n_sub, theta0_rad,
                        pol)


def transmittance(lam_um, t_um, n_layers, n_inc=1.0, n_sub=1.0,
                  theta0_rad: float = 0.0, pol: str = "s", n_exit=None,
                  sub_thickness_um=None):
    """Intensity transmittance T(lam) (n_exit, sub_thickness_um: see
    `stack_rt`)."""
    return stack_rt(lam_um, t_um, n_layers, n_inc, n_sub,
                    theta0_rad, pol, n_exit, sub_thickness_um)[1]


def reflectance(lam_um, t_um, n_layers, n_inc=1.0, n_sub=1.0,
                theta0_rad: float = 0.0, pol: str = "s", n_exit=None,
                sub_thickness_um=None):
    """Intensity reflectance R(lam) (n_exit, sub_thickness_um: see
    `stack_rt`)."""
    return stack_rt(lam_um, t_um, n_layers, n_inc, n_sub,
                    theta0_rad, pol, n_exit, sub_thickness_um)[0]


# ------------------------- linear merits --------------------------


def _check_band_args(**kw):
    for name, v in kw.items():
        if not (np.isfinite(float(v)) and float(v) >= 0.0):
            raise ValueError(f"{name} must be finite and >= 0; got {v!r} "
                             "(a negative width makes the bands overlap)")


def _edge_tol(lam):
    """Tolerance for a wavelength that lies on a band edge (new in
    0.11.0). An edge such as c + h is computed in binary floating point
    and can land just inside a grid point that is exactly on the edge
    in decimal: 0.5325 + 0.0125 is 0.5449999999999999, so the grid
    point 0.545 fell outside the stop band [0.52, 0.545]. Up to 0.10.x
    such edge points were dropped from one side of a band and kept on
    the other. A point within 1e-9 of the largest wavelength (1e-9 um
    for wavelengths near 1 um) of an edge now counts as on it; a real
    grid is far coarser than that."""
    return 1e-9 * float(np.max(np.abs(lam))) if lam.size else 0.0


def notch_weights(lam_um, center_um: float, half_um: float,
                  guard_um: float) -> tuple:
    """Weights (w, const) of the fluorescence-rejection notch merit

        J = 0.5 * mean_pass T + 0.5 * mean_stop (1 - T) = w . T + const

    with stop band [c-h, c+h] and pass bands outside the guard
    [c-h-g, c+h+g] (Mahim et al., IEEE Sensors J., 2026, Eq. 2).
    half_um and guard_um must be >= 0 (checked since 0.9.0). A
    wavelength exactly on an edge belongs to the band, also when
    floating-point round-off puts the computed edge just past it
    (since 0.11.0; see `_edge_tol`).
    """
    _check_band_args(half_um=half_um, guard_um=guard_um)
    lam = np.asarray(lam_um, dtype=float)
    tol = _edge_tol(lam)
    stop = (lam >= center_um - half_um - tol) & \
           (lam <= center_um + half_um + tol)
    pas = (lam <= center_um - half_um - guard_um + tol) | \
          (lam >= center_um + half_um + guard_um - tol)
    if not stop.any() or not pas.any():
        raise ValueError("band layout leaves an empty stop or pass band "
                         "on this wavelength grid")
    w = 0.5 * pas / pas.sum() - 0.5 * stop / stop.sum()
    return w, 0.5


def bandpass_weights(lam_um, lo_um: float, hi_um: float,
                     guard_um: float) -> tuple:
    """Weights of the dual bandpass merit: J = 0.5 mean_in T
    + 0.5 mean_out (1 - T), pass band [lo, hi], rejection outside the
    guard. guard_um must be >= 0 (checked since 0.9.0). Edge points
    belong to their band (since 0.11.0, as in `notch_weights`)."""
    _check_band_args(guard_um=guard_um)
    lam = np.asarray(lam_um, dtype=float)
    tol = _edge_tol(lam)
    inside = (lam >= lo_um - tol) & (lam <= hi_um + tol)
    outside = (lam <= lo_um - guard_um + tol) | \
        (lam >= hi_um + guard_um - tol)
    if not inside.any() or not outside.any():
        raise ValueError("band layout leaves an empty band on this grid")
    w = 0.5 * inside / inside.sum() - 0.5 * outside / outside.sum()
    return w, 0.5


def weights_from_reflectance(weights_R, const=0.0):
    """Convert a reflectance-based linear merit into transmittance
    weights, so mirror and high-reflector merits run through the same
    exact adjoint (new in v0.3).

    For a LOSSLESS stack R = 1 - T exactly (asserted to machine
    precision in the solver tests), so

        J = w_R . R + c = (-w_R) . T + (c + sum w_R)

    is an identity, and the returned pair (w_T, const_T) feeds
    `merit`, `merit_and_grad`, `inverse_design` and `robustify`
    unchanged. For an ABSORBING stack R = 1 - T - A and the identity
    fails by exactly w_R . A -- which is why the lossless hand-adjoint
    regime is the scope here, matching `fabtwin.adjoint`.
    """
    w = np.asarray(weights_R, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights_R must be a 1-D weight vector")
    return -w, float(const) + float(w.sum())


def merit(T, weights, const):
    """J = w . T + const for one spectrum or a batch (..., L)."""
    T = np.asarray(T)
    return T @ np.asarray(weights) + const
