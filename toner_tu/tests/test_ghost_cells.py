import numpy as np
import pytest

from toner_tu.boundary_conditions import (
    ChannelBoundaryConditions,
    add_ghost_cells,
)


@pytest.fixture
def boundary():
    return ChannelBoundaryConditions(
        rho_in=2.0,
        vx_in=1.0,
        vy_in=0.0,
    )


def test_ghost_cells_shape(boundary):
    field = np.ones((4, 6))

    result = add_ghost_cells(field, boundary, "rho")

    assert result.shape == (6, 8)


def test_density_inlet(boundary):
    field = np.full((3, 4), 0.5)

    result = add_ghost_cells(field, boundary, "rho")

    assert np.allclose(result[1:-1, 0], 3.5)


def test_velocity_inlet(boundary):
    field = np.full((3, 4), 0.2)

    result = add_ghost_cells(field, boundary, "vx")

    assert np.allclose(result[1:-1, 0], 1.8)


def test_outlet_zero_gradient(boundary):
    field = np.arange(12, dtype=float).reshape(3, 4)

    result = add_ghost_cells(field, boundary, "rho")

    assert np.allclose(result[1:-1, -1], field[:, -1])


def test_wall_density_zero_gradient(boundary):
    field = np.arange(12, dtype=float).reshape(3, 4)

    result = add_ghost_cells(field, boundary, "rho")

    assert np.allclose(result[0, 1:-1], field[0, :])
    assert np.allclose(result[-1, 1:-1], field[-1, :])


def test_wall_normal_velocity_reflection(boundary):
    field = np.full((3, 4), 0.7)

    result = add_ghost_cells(field, boundary, "vy")

    assert np.allclose(result[0, 1:-1], -0.7)
    assert np.allclose(result[-1, 1:-1], -0.7)


def test_unknown_field_raises_error(boundary):
    field = np.ones((3, 4))

    with pytest.raises(ValueError):
        add_ghost_cells(field, boundary, "temperature")