
import numpy as np
import pytest

from toner_tu.fields import Grid
from toner_tu.operators import (
    gradient,
    divergence,
    laplacian,
    advective_derivative,
)


@pytest.fixture
def grid():
    return Grid(nx=100, ny=100, lx=2 * np.pi, ly=2 * np.pi)


@pytest.fixture
def coordinates(grid):
    return grid.coordinates()


def test_gradient(grid, coordinates):
    x, y = coordinates
    field = np.sin(x) * np.cos(y)

    dfdx, dfdy = gradient(field, grid)

    exact_x = np.cos(x) * np.cos(y)
    exact_y = -np.sin(x) * np.sin(y)

    assert np.allclose(dfdx, exact_x, atol=0.002)
    assert np.allclose(dfdy, exact_y, atol=0.002)


def test_divergence(grid, coordinates):
    x, y = coordinates

    vx = np.sin(x) * np.cos(y)
    vy = np.cos(x) * np.sin(y)

    result = divergence(vx, vy, grid)

    exact = 2.0 * np.cos(x) * np.cos(y)

    assert np.allclose(result, exact, atol=0.002)


def test_laplacian(grid, coordinates):
    x, y = coordinates

    field = np.sin(x) * np.cos(y)
    result = laplacian(field, grid)

    exact = -2.0 * np.sin(x) * np.cos(y)

    assert np.allclose(result, exact, atol=0.005)


def test_advective_derivative(grid, coordinates):
    x, y = coordinates

    vx = np.sin(x)
    vy = np.sin(y)

    adv_x, adv_y = advective_derivative(vx, vy, grid)

    exact_x = np.sin(x) * np.cos(x)
    exact_y = np.sin(y) * np.cos(y)

    assert np.allclose(adv_x, exact_x, atol=0.002)
    assert np.allclose(adv_y, exact_y, atol=0.002)
