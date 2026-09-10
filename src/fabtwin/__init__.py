"""fabtwin: learned generative process twins for yield-aware inverse
design of multilayer optics.

Generalized library form of the FabGAN-ID framework (T. M. Mahim,
M. N. Islam, M. M. Rahman, A. S. M. Mohsin, "FabGAN-ID: Learning the
Fabrication Process for Yield-Aware Inverse Design of Multilayer
Photonic Sensor Filters", IEEE Sensors Journal, 2026): exact
transfer-matrix physics with a hand-derived discrete adjoint (no
autodiff framework required), a parametric six-mechanism virtual
deposition process, Gaussian and learned (conditional WGAN-GP, the
optional [twin] JAX extra) process twins on a common error
parameterization, probe-seeded adjoint inverse design, and pathwise
CVaR robustification through twin and physics at once. The guiding
principle of the paper is the guiding principle of the package:
learn what cannot be simulated; differentiate what can.
"""
from .materials import (SI3N4_LUKE2015, SIO2_MALITSON1965,
                        SellmeierMaterial, dispersion_shape, layer_index,
                        sellmeier)
from .tmm import (bandpass_weights, merit, notch_weights, reflectance,
                  stack_rt, transmittance)
from .adjoint import merit_and_grad, transmittance_and_grads
from .process import PAPER_PROCESS, DepositionProcess
from .twins import GaussianTwin, apply_errors, errors_from_traces
from .risk import (cvar, evaluate_under_process, pass_fail,
                   tail_statistics, yield_fraction)
from .design import DesignBox, adam_ascent, inverse_design, random_search
from .robust import cvar_objective_and_grad, robustify

__version__ = "0.1.0"
__all__ = [
    "SellmeierMaterial", "SI3N4_LUKE2015", "SIO2_MALITSON1965",
    "sellmeier", "dispersion_shape", "layer_index",
    "transmittance", "reflectance", "stack_rt", "merit",
    "notch_weights", "bandpass_weights",
    "transmittance_and_grads", "merit_and_grad",
    "DepositionProcess", "PAPER_PROCESS",
    "GaussianTwin", "errors_from_traces", "apply_errors",
    "cvar", "tail_statistics", "pass_fail", "yield_fraction",
    "evaluate_under_process",
    "DesignBox", "inverse_design", "random_search", "adam_ascent",
    "cvar_objective_and_grad", "robustify",
]
