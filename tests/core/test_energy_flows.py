"""EnergyFlows arithmetic (CONTRACTS section 2)."""

from __future__ import annotations

import pytest

from thermosim.core.types import EnergyFlows


def _flows(**kwargs: float) -> EnergyFlows:
    return EnergyFlows(**kwargs)


def test_defaults_are_zero() -> None:
    e = EnergyFlows()
    assert e.total_in() == 0.0
    assert e.total_loss() == 0.0
    assert e.total_stored() == 0.0
    assert e.balance_error() == 0.0


def test_totals_sum_the_right_fields() -> None:
    e = _flows(
        source_liquid=10.0,
        source_shell=2.0,
        loss_shell_outer=1.0,
        loss_surface_convrad=0.5,
        loss_surface_evap=0.25,
        loss_lid_outer=0.125,
        loss_dry_wall_open=0.0625,
        stored_liquid=7.0,
        stored_shell=1.0,
        stored_lid_headspace=-0.5,
    )
    assert e.total_in() == pytest.approx(12.0)
    assert e.total_loss() == pytest.approx(1.0 + 0.5 + 0.25 + 0.125 + 0.0625)
    assert e.total_stored() == pytest.approx(7.5)
    assert e.balance_error() == pytest.approx(12.0 - 1.9375 - 7.5)


def test_add_accumulates_every_field_in_place() -> None:
    a = _flows(source_liquid=1.0, loss_shell_outer=2.0, stored_liquid=3.0)
    b = _flows(source_liquid=10.0, loss_surface_evap=4.0, stored_shell=-1.0)
    returned = a.add(b)
    assert returned is None
    assert a.source_liquid == pytest.approx(11.0)
    assert a.loss_shell_outer == pytest.approx(2.0)
    assert a.loss_surface_evap == pytest.approx(4.0)
    assert a.stored_liquid == pytest.approx(3.0)
    assert a.stored_shell == pytest.approx(-1.0)
    # The other operand is not modified.
    assert b.source_liquid == pytest.approx(10.0)


def test_balance_error_is_zero_for_a_closed_budget() -> None:
    e = _flows(source_liquid=100.0, loss_shell_outer=30.0, stored_liquid=70.0)
    assert e.balance_error() == pytest.approx(0.0, abs=1e-12)
