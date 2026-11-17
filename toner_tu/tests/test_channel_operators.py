
import numpy as np
import pytest

from toner_tu.fields import Grid
from toner_tu.channel_operators import (
    gradient,
    divergence,
    laplacian,
    advective_derivative,
)


@pytest.fixture
def grid():
    return Grid(
        nx=64,
        ny=64,
        lx=10.0,
        ly=10.0,
    )


def test_gradient_of_linear_function(grid):
    x, y = grid.coordinates()

    field = 2.0 * x + 3.0 * y

    dx, dy = gradient(field, grid)

    assert np.allclose(dx, 2.0)
    assert np.allclose(dy, 3.0)


def test_divergence_of_linear_vector_field(grid):
    x, y = grid.coordinates()

    vx = 2.0 * x
    vy = 3.0 * y

    result = divergence(vx, vy, grid)

    assert np.allclose(result, 5.0)


def test_laplacian_of_quadratic_function(grid):
    x, y = grid.coordinates()

    field = x**2 + y**2

    result = laplacian(field, grid)

    # The exact Laplacian is 2 + 2 = 4.
    assert np.allclose(result[2:-2, 2:-2], 4.0)


def test_advective_derivative(grid):
    x, y = grid.coordinates()

    vx = 2.0 * x
    vy = 3.0 * y

    adv_x, adv_y = advective_derivative(vx, vy, grid)

    expected_x = 4.0 * x
    expected_y = 9.0 * y

    assert np.allclose(adv_x, expected_x)
    assert np.allclose(adv_y, expected_y)


def test_invalid_shape(grid):
    field = np.zeros((10, 10))

    with pytest.raises(ValueError):
        gradient(field, grid)
