"""One simulation step: orchestration of physics, sources and solver (CONTRACTS section 12).

WP00 stub, implemented by WP09.
"""

from __future__ import annotations

from thermosim.config.schema import Scenario
from thermosim.core.protocols import CommandSource
from thermosim.core.types import Domain, Sample, Snapshot, StopReason, ThermalState


class Model:
    state: ThermalState
    domain: Domain

    def __init__(self, scenario: Scenario, command: CommandSource) -> None:
        raise NotImplementedError

    def advance(self, until_t: float) -> list[Sample | Snapshot]:
        """Step until state.t >= until_t; return the samples and snapshots produced."""
        raise NotImplementedError

    @property
    def stop_reason(self) -> StopReason | None:
        raise NotImplementedError
