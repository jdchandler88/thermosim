"""Independent validation aids: analytic references and the lumped (0-D) reference model."""

from thermosim.validation.analytic import (
    gaussian_pulse_3d,
    lumped_newton,
    semi_infinite_convection,
    semi_infinite_step,
)
from thermosim.validation.lumped import LumpedModel, LumpedResult, quick_estimate

__all__ = [
    "LumpedModel",
    "LumpedResult",
    "gaussian_pulse_3d",
    "lumped_newton",
    "quick_estimate",
    "semi_infinite_convection",
    "semi_infinite_step",
]
