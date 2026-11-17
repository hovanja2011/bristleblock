
import numpy as np
import pytest

from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.equations import compute_rhs


@pytest.fixture
def grid():
    return Grid(nx=64, ny=64, lx=20.0, ly=20.0)


@pytest.fixture
def params():
    return TonerTuParameters(
        alpha=1.0,
        beta=1.0,
        sigma=1.0,
        diffusion=0.1,
        lambda_=1.0,
    )


def test_uniform_stationary_state(grid, params):
    rho = np.ones((grid.ny, grid.nx))
    vx = np.zeros_like(rho)
    vy = np.zeros_like(rho)

    fields = Fields(rho, vx, vy)

    rhs = compute_rhs(fields, grid, params)

    assert np.allclose(rhs.rho, 0.0)
    assert np.allclose(rhs.vx, 0.0)
    assert np.allclose(rhs.vy, 0.0)


def test_uniform_ordered_state(grid, params):
    rho = np.ones((grid.ny, grid.nx))

    speed = params.ordered_speed

    vx = np.full_like(rho, speed)
    vy = np.zeros_like(rho)

    fields = Fields(rho, vx, vy)

    rhs = compute_rhs(fields, grid, params)

    assert np.allclose(rhs.rho, 0.0)
    assert np.allclose(rhs.vx, 0.0, atol=1e-12)
    assert np.allclose(rhs.vy, 0.0, atol=1e-12)


def test_mass_conservation(grid, params):
    rng = np.random.default_rng(42)

    rho = 1.0 + 0.1 * rng.random((grid.ny, grid.nx))
    vx = rng.normal(size=rho.shape)
    vy = rng.normal(size=rho.shape)

    fields = Fields(rho, vx, vy)

    rhs = compute_rhs(fields, grid, params)

    total_mass_derivative = np.sum(rhs.rho) * grid.dx * grid.dy

    assert total_mass_derivative == pytest.approx(0.0, abs=1e-10)


def test_velocity_growth_below_ordered_speed(grid, params):
    rho = np.ones((grid.ny, grid.nx))

    vx = np.full_like(rho, 0.5)
    vy = np.zeros_like(rho)

    fields = Fields(rho, vx, vy)

    rhs = compute_rhs(fields, grid, params)

    assert np.all(rhs.vx > 0.0)
    assert np.allclose(rhs.vy, 0.0)
