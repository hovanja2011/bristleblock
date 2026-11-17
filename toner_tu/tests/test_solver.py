
import numpy as np
import pytest

from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.solver import step, integrate


@pytest.fixture
def grid():
    return Grid(nx=32, ny=32, lx=10.0, ly=10.0)


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

    result = step(fields, grid, params, dt=0.01)

    assert np.allclose(result.rho, rho)
    assert np.allclose(result.vx, vx)
    assert np.allclose(result.vy, vy)


def test_uniform_ordered_state(grid, params):
    rho = np.ones((grid.ny, grid.nx))
    vx = np.full_like(rho, params.ordered_speed)
    vy = np.zeros_like(rho)

    fields = Fields(rho, vx, vy)

    result = integrate(fields, grid, params, dt=0.01, n_steps=100)

    assert np.allclose(result.rho, rho)
    assert np.allclose(result.vx, vx, atol=1e-10)
    assert np.allclose(result.vy, vy, atol=1e-10)


def test_mass_conservation(grid, params):
    rng = np.random.default_rng(42)

    rho = 1.0 + 0.1 * rng.random((grid.ny, grid.nx))
    vx = rng.normal(scale=0.1, size=rho.shape)
    vy = rng.normal(scale=0.1, size=rho.shape)

    fields = Fields(rho, vx, vy)

    initial_mass = np.sum(fields.rho) * grid.dx * grid.dy

    result = integrate(fields, grid, params, dt=0.001, n_steps=10)

    final_mass = np.sum(result.rho) * grid.dx * grid.dy

    assert final_mass == pytest.approx(initial_mass, abs=1e-10)


def test_input_fields_are_not_modified(grid, params):
    rho = np.ones((grid.ny, grid.nx))
    vx = np.full_like(rho, 0.2)
    vy = np.zeros_like(rho)

    fields = Fields(rho, vx, vy)
    original_rho = fields.rho.copy()
    original_vx = fields.vx.copy()

    integrate(fields, grid, params, dt=0.001, n_steps=5)

    assert np.array_equal(fields.rho, original_rho)
    assert np.array_equal(fields.vx, original_vx)


def test_invalid_time_step(grid, params):
    fields = Fields(
        np.ones((grid.ny, grid.nx)),
        np.zeros((grid.ny, grid.nx)),
        np.zeros((grid.ny, grid.nx)),
    )

    with pytest.raises(ValueError):
        step(fields, grid, params, dt=0.0)
