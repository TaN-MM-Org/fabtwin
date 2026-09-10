"""Material dispersion: a general Sellmeier engine with cited built-ins.

Every built-in coefficient set carries its literature source and its
stated validity range, and evaluation OUTSIDE that range is refused
rather than silently extrapolated -- a Sellmeier fit knows nothing
about the material beyond the data it was fitted to.

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

__all__ = ["SellmeierMaterial", "SI3N4_LUKE2015", "SIO2_MALITSON1965",
           "dispersion_shape", "layer_index", "sellmeier"]


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


def dispersion_shape(lam_um, lam0_um, material: SellmeierMaterial = SI3N4_LUKE2015):
    """Dispersion shape S(lam) = n(lam)/n(lam0), exactly 1 at lam0."""
    return material.n(lam_um) / float(material.n(lam0_um))


def layer_index(n_at_lam0, lam_um, lam0_um,
                material: SellmeierMaterial = SI3N4_LUKE2015):
    """Variable-index layer: n_i(lam) = n_i(lam0) * S(lam).

    n_at_lam0 : scalar or (N,) prescribed indices at lam0.
    Returns (L,) or (N, L).
    """
    n0 = np.asarray(n_at_lam0, dtype=float)
    S = dispersion_shape(lam_um, lam0_um, material)
    if n0.ndim == 0:
        return n0 * S
    return n0[:, None] * S[None, :]
