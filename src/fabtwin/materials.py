"""Material dispersion: a Sellmeier engine with cited built-ins, and
measured dispersion tables for laboratory data.

Every built-in coefficient set carries its literature source and its
stated validity range, and evaluation OUTSIDE that range is refused
rather than silently extrapolated -- a Sellmeier fit knows nothing
about the material beyond the data it was fitted to.

`TabulatedMaterial` (new in v0.3) is the experimental counterpart: a
measured n(lam) table -- ellipsometry output, a vendor datasheet, a
refractiveindex.info tabulation -- used directly, with no Sellmeier
fit required. Interpolation is shape-preserving PCHIP (piecewise
cubic Hermite; no overshoot between measured points), evaluation
outside the tabulated range is refused exactly as for Sellmeier
built-ins, and a mandatory `reference` names where the table comes
from. An optional extinction column k feeds the absorbing FORWARD
solver through `.nk()`; `.n()` refuses a table with nonzero k rather
than silently dropping the absorption, because the hand adjoint's
regime is lossless. Anchors asserted in the tests rather than
stated: the interpolant reproduces the tabulated nodes exactly, and
on a table sampled from a cited Sellmeier built-in it agrees with
the independent closed form between the nodes.

Built-ins (refractiveindex.info open database, CC0):

* Si3N4: K. Luke et al., Opt. Lett. 40, 4823 (2015), Sellmeier
  "formula 1", valid 0.310-5.504 um.
* SiO2 (fused silica): I. H. Malitson, J. Opt. Soc. Am. 55, 1205
  (1965), valid 0.21-6.7 um.

The variable-index layer model follows the single-material SiNx
platform of Yesilyurt et al., Nanophotonics 12, 993 (2023): a layer
with prescribed index n(lam0) at the reference wavelength carries the
Si3N4 dispersion *shape* scaled to that index,
n_i(lam) = n_i(lam0) * n_Si3N4(lam)/n_Si3N4(lam0). `dispersion_shape`
returns that shape for any registered material, so the same
construction ports to other material systems. Anchors asserted in the
tests rather than stated: the engine reproduces the closed-form
Sellmeier evaluation exactly, the shape equals 1 at the reference
wavelength exactly, and out-of-range evaluation raises.
"""
from __future__ import annotations

import dataclasses

import numpy as np

__all__ = ["SellmeierMaterial", "TabulatedMaterial", "SI3N4_LUKE2015",
           "SIO2_MALITSON1965", "dispersion_shape", "layer_index",
           "sellmeier"]


@dataclasses.dataclass(frozen=True)
class SellmeierMaterial:
    """A cited Sellmeier model: n^2 = 1 + sum_i c1_i lam^2/(lam^2 - c2_i^2)
    (wavelengths in um), valid only on [lam_min, lam_max]."""

    name: str
    terms: tuple                 # ((c1, c2), ...)
    lam_min: float               # um
    lam_max: float               # um
    reference: str

    def n(self, lam_um):
        lam = np.asarray(lam_um, dtype=float)
        if np.any(lam < self.lam_min) or np.any(lam > self.lam_max):
            raise ValueError(
                f"{self.name}: wavelength outside the fitted validity "
                f"range [{self.lam_min}, {self.lam_max}] um "
                f"({self.reference}); a Sellmeier fit must not be "
                "extrapolated silently")
        return sellmeier(lam, self.terms)


def sellmeier(lam_um, terms):
    """Bare Sellmeier evaluation (no range check): n(lam_um)."""
    lam2 = np.asarray(lam_um, dtype=float) ** 2
    n2 = 1.0 + sum(c1 * lam2 / (lam2 - c2 ** 2) for c1, c2 in terms)
    return np.sqrt(n2)


SI3N4_LUKE2015 = SellmeierMaterial(
    name="Si3N4 (Luke 2015)",
    terms=((3.0249, 0.1353406), (40314.0, 1239.842)),
    lam_min=0.310, lam_max=5.504,
    reference="K. Luke et al., Opt. Lett. 40, 4823 (2015); "
              "refractiveindex.info (CC0)")

SIO2_MALITSON1965 = SellmeierMaterial(
    name="fused silica (Malitson 1965)",
    terms=((0.6961663, 0.0684043), (0.4079426, 0.1162414),
           (0.8974794, 9.896161)),
    lam_min=0.21, lam_max=6.7,
    reference="I. H. Malitson, J. Opt. Soc. Am. 55, 1205 (1965); "
              "refractiveindex.info (CC0)")


@dataclasses.dataclass(frozen=True, eq=False)
class TabulatedMaterial:
    """A measured dispersion table: n (and optionally k) at tabulated
    wavelengths, interpolated by shape-preserving PCHIP and refused
    outside the tabulated range.

    lam_um : (P,) strictly increasing tabulated wavelengths (um).
    n_table : (P,) real refractive index at those wavelengths.
    k_table : optional (P,) extinction coefficient, k >= 0 (n + i k
        the package's absorbing convention). Omit for a lossless
        table.
    reference : where the table comes from (your ellipsometry run, a
        datasheet, refractiveindex.info page). Required, on purpose.

    `.n(lam)` is the same interface as `SellmeierMaterial.n`, so a
    tabulated material drops into `dispersion_shape`, `layer_index`
    and everything built on them; it refuses an absorbing table (the
    design path is lossless), which `.nk(lam)` serves instead for the
    forward solver.
    """

    name: str
    lam_um: tuple
    n_table: tuple
    k_table: tuple = None
    reference: str = ""

    def __post_init__(self):
        lam = np.asarray(self.lam_um, dtype=float)
        n = np.asarray(self.n_table, dtype=float)
        if lam.ndim != 1 or lam.size < 2 or n.shape != lam.shape:
            raise ValueError("lam_um and n_table must be 1-D arrays "
                             "of equal length >= 2")
        if not np.all(np.isfinite(lam)) or not np.all(np.isfinite(n)):
            raise ValueError("tabulated values must be finite")
        if np.any(np.diff(lam) <= 0.0):
            raise ValueError("lam_um must be strictly increasing")
        if np.any(n <= 0.0):
            raise ValueError("tabulated n must be positive")
        if self.k_table is not None:
            k = np.asarray(self.k_table, dtype=float)
            if k.shape != lam.shape or not np.all(np.isfinite(k)):
                raise ValueError("k_table must match lam_um and be "
                                 "finite")
            if np.any(k < 0.0):
                raise ValueError("k must be >= 0 (gain is out of "
                                 "scope; the convention is n + i k)")
        else:
            k = None
        if not isinstance(self.reference, str) or not self.reference.strip():
            raise ValueError(
                "TabulatedMaterial requires a non-empty `reference` "
                "naming the source of the table (your measurement, a "
                "datasheet, a database entry)")
        from scipy.interpolate import PchipInterpolator
        object.__setattr__(self, "lam_um", lam)
        object.__setattr__(self, "n_table", n)
        object.__setattr__(self, "k_table", k)
        object.__setattr__(self, "_pn", PchipInterpolator(lam, n))
        object.__setattr__(self, "_pk",
                           PchipInterpolator(lam, k) if k is not None
                           else None)

    @property
    def lam_min(self):
        return float(self.lam_um[0])

    @property
    def lam_max(self):
        return float(self.lam_um[-1])

    def _check_range(self, lam):
        if np.any(lam < self.lam_min) or np.any(lam > self.lam_max):
            raise ValueError(
                f"{self.name}: wavelength outside the tabulated range "
                f"[{self.lam_min}, {self.lam_max}] um "
                f"({self.reference}); a measured table must not be "
                "extrapolated silently")

    def is_absorbing(self):
        return self.k_table is not None and bool(np.any(self.k_table > 0.0))

    def n(self, lam_um):
        """Real index n(lam) by PCHIP interpolation (exact at the
        tabulated nodes). Refuses an absorbing table rather than
        silently dropping k -- use `.nk()` with the forward solver."""
        if self.is_absorbing():
            raise ValueError(
                f"{self.name} has nonzero extinction k; the lossless "
                "design path (dispersion shapes, the hand adjoint) "
                "cannot honestly represent it. Use material.nk() "
                "with the fabtwin.tmm forward solver instead")
        lam = np.asarray(lam_um, dtype=float)
        self._check_range(lam)
        return self._pn(lam)

    def nk(self, lam_um):
        """Complex index n + i k on the tabulated range (for the
        absorbing forward solver of `fabtwin.tmm`)."""
        lam = np.asarray(lam_um, dtype=float)
        self._check_range(lam)
        kk = self._pk(lam) if self._pk is not None else 0.0
        return self._pn(lam) + 1j * kk


def dispersion_shape(lam_um, lam0_um, material=SI3N4_LUKE2015):
    """Dispersion shape S(lam) = n(lam)/n(lam0), exactly 1 at lam0.
    `material` is anything with an `.n(lam_um)` method: a built-in or
    user SellmeierMaterial, or a (lossless) TabulatedMaterial."""
    return material.n(lam_um) / float(
        np.asarray(material.n(lam0_um)).reshape(-1)[0])


def layer_index(n_at_lam0, lam_um, lam0_um,
                material=SI3N4_LUKE2015):
    """Variable-index layer: n_i(lam) = n_i(lam0) * S(lam).

    n_at_lam0 : scalar or (N,) prescribed indices at lam0.
    Returns (L,) or (N, L).
    """
    n0 = np.asarray(n_at_lam0, dtype=float)
    S = dispersion_shape(lam_um, lam0_um, material)
    if n0.ndim == 0:
        return n0 * S
    return n0[:, None] * S[None, :]
