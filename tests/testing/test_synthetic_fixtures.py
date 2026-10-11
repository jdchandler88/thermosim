"""uniform_coefficients and ConstantProps (WP00 fixtures used by WP03 and others)."""

from __future__ import annotations

import numpy as np
import pytest

from thermosim.core.types import Domain
from thermosim.physics.properties import AirProps
from thermosim.testing.synthetic import ConstantProps, uniform_coefficients


def test_uniform_coefficients_shapes_and_values(box: Domain) -> None:
    c = uniform_coefficients(box, k=0.5, rho_c=4.0e6, h_film=10.0, h_out=5.0, h_surface=3.0)
    shape = box.grid.shape
    n = box.shell.n
    ns = len(box.grid.surface_cells)
    assert c.k_liquid.shape == shape and np.all(c.k_liquid == 0.5)
    assert c.rho_c_liquid.shape == shape and np.all(c.rho_c_liquid == 4.0e6)
    assert c.h_film.shape == (len(box.shell.couple_shell),) and np.all(c.h_film == 10.0)
    assert c.h_out_shell.shape == (n,) and np.all(c.h_out_shell == 5.0)
    assert c.h_dry.shape == (n,)
    assert c.h_surface_open.shape == (ns,) and np.all(c.h_surface_open == 3.0)
    assert c.h_surface_covered.shape == (ns,)
    assert c.q_evap_open.shape == (ns,) and np.all(c.q_evap_open == 0.0)
    assert c.q_cells.shape == shape and np.all(c.q_cells == 0.0)
    assert c.q_shell.shape == (n,) and np.all(c.q_shell == 0.0)
    assert c.T_amb == pytest.approx(293.15)


def test_uniform_coefficients_defaults_have_no_losses(box: Domain) -> None:
    c = uniform_coefficients(box)
    assert np.all(c.h_film == 0.0)
    assert np.all(c.h_out_shell == 0.0)
    assert c.h_lid_inner == 0.0 and c.h_lid_outer == 0.0


def test_constant_props_scalar_and_array_inputs() -> None:
    assert ConstantProps.wort_density(293.15, 1.0) == pytest.approx(998.2)
    arr = ConstantProps.wort_density(np.full((2, 3), 293.15), 1.0)
    assert isinstance(arr, np.ndarray) and arr.shape == (2, 3)
    assert np.all(arr == 998.2)
    assert ConstantProps.latent_heat_vaporization(373.15) == pytest.approx(2.454e6)


def test_constant_props_air_is_airprops() -> None:
    air = ConstantProps.air_props(293.15, 101325.0)
    assert isinstance(air, AirProps)
    assert air.prandtl == pytest.approx(0.71, rel=0.02)


def test_sg_plato_roundtrip() -> None:
    sg = 1.05
    assert ConstantProps.plato_to_sg(ConstantProps.sg_to_plato(sg)) == pytest.approx(sg)
    assert ConstantProps.sg_to_plato(1.0) == pytest.approx(0.0)
