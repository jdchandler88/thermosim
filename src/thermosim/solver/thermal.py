"""Finite-volume explicit thermal solver (CONTRACTS.md section 10).

WP00 stub, implemented by WP06.
"""

from __future__ import annotations

from thermosim.core.types import Domain, EnergyFlows, StepCoefficients, ThermalState


class ThermalSolver:
    def __init__(self, domain: Domain) -> None:
        """Precomputes index arrays and triggers the Numba compile."""
        raise NotImplementedError

    def initial_state(self, T0_liquid: float, T0_shell: float, T_amb: float) -> ThermalState:
        """Uniform initial temperatures (K)."""
        raise NotImplementedError

    def stable_dt(self, coeffs: StepCoefficients) -> float:
        """Liquid stability limit times 0.9, s."""
        raise NotImplementedError

    def step(self, state: ThermalState, coeffs: StepCoefficients, dt: float) -> EnergyFlows:
        """Advance `state` IN PLACE by dt seconds (state.t += dt). Shell, lid and headspace may
        substep internally. Raises SolverError if dt > stable_dt or any NaN appears.
        Returns the energy (J) that flowed during the step."""
        raise NotImplementedError

    def internal_energy(self, state: ThermalState, coeffs: StepCoefficients) -> float:
        """Total internal energy of the state, J, with reference 0 K."""
        raise NotImplementedError
