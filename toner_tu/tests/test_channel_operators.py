
import numpy as np
import pytest

from toner_tu.fields import Grid
from toner_tu.boundary_conditions import ChannelBoundaryConditions
from toner_tu.channel_operators import (
    gradient,
    laplacian,
    advective_derivative,
)
from toner_tu.channel_operators import advective_derivative_upwind


@pytest.fixture
def grid():
    return Grid(
        nx=64,
        ny=64,
        lx=10.0,
        ly=10.0,
    )


@pytest.fixture
def boundary():
    return ChannelBoundaryConditions(
        rho_in=0.0,
        vx_in=0.0,
        vy_in=0.0,
    )


def test_gradient_of_linear_function(grid, boundary):
    x, y = grid.coordinates()
    field = 2.0 * x + 3.0 * y

    dx, dy = gradient(field, grid, boundary, "rho")

    # Проверяем внутренние ячейки, не затрагивая границы.
    assert np.allclose(dx[:, 1:-1], 2.0)
    assert np.allclose(dy[1:-1, :], 3.0)


def test_laplacian_of_quadratic_function(grid, boundary):
    x, y = grid.coordinates()
    field = x**2 + y**2

    result = laplacian(field, grid, boundary, "rho")

    # Точный лапласиан равен 2 + 2 = 4.
    assert np.allclose(result[2:-2, 2:-2], 4.0)


def test_advective_derivative(grid, boundary):
    x, y = grid.coordinates()

    vx = 2.0 * x
    vy = 3.0 * y

    adv_x, adv_y = advective_derivative(
        vx, vy, grid, boundary
    )

    expected_x = 4.0 * x
    expected_y = 9.0 * y

    assert np.allclose(adv_x[1:-1, 1:-1],
                       expected_x[1:-1, 1:-1])
    assert np.allclose(adv_y[1:-1, 1:-1],
                       expected_y[1:-1, 1:-1])


def test_invalid_shape(grid, boundary):
    field = np.zeros((10, 10))

    with pytest.raises(ValueError):
        gradient(field, grid, boundary, "rho")



def test_advective_derivative_upwind_of_linear_fields(grid, boundary):
    x, y = grid.coordinates()

    vx = 2.0 * x
    vy = 3.0 * y

    adv_x, adv_y = advective_derivative_upwind(
        vx, vy, grid, boundary
    )

    expected_x = 4.0 * x
    expected_y = 9.0 * y

    assert np.allclose(
        adv_x[2:-2, 2:-2],
        expected_x[2:-2, 2:-2],
    )
    assert np.allclose(
        adv_y[2:-2, 2:-2],
        expected_y[2:-2, 2:-2],
    )
