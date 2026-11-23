import numpy as np
import pytest

from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.solver import step, integrate, max_stable_dt


@pytest.fixture
def grid():
    return Grid(nx=32, ny=32, lx=10.0, ly=10.0)


@pytest.fixture
def params():
    return TonerTuParameters(alpha=1.0, beta=1.0, sigma=1.0, diffusion=0.1, lambda_=1.0)


def test_uniform_stationary_state(grid, params):
    fields = Fields(np.ones((grid.ny, grid.nx)), np.zeros((grid.ny, grid.nx)), np.zeros((grid.ny, grid.nx)))
    result = step(fields, grid, params, dt=0.01)
    assert np.allclose(result.rho, fields.rho)
    assert np.allclose(result.vx, fields.vx)
    assert np.allclose(result.vy, fields.vy)


def test_uniform_ordered_state(grid, params):
    rho = np.ones((grid.ny, grid.nx))
    fields = Fields(rho, np.full_like(rho, params.ordered_speed), np.zeros_like(rho))
    result = integrate(fields, grid, params, dt=0.01, n_steps=100)
    assert np.allclose(result.rho, rho)
    assert np.allclose(result.vx, fields.vx, atol=1e-10)
    assert np.allclose(result.vy, fields.vy, atol=1e-10)


def test_mass_conservation(grid, params):
    rng = np.random.default_rng(42)
    rho = 1.0 + 0.1*rng.random((grid.ny, grid.nx))
    fields = Fields(rho, rng.normal(scale=0.1, size=rho.shape), rng.normal(scale=0.1, size=rho.shape))
    initial_mass = np.sum(fields.rho)*grid.dx*grid.dy
    result = integrate(fields, grid, params, dt=0.001, n_steps=10)
    final_mass = np.sum(result.rho)*grid.dx*grid.dy
    assert final_mass == pytest.approx(initial_mass, abs=1e-10)


def test_input_fields_are_not_modified(grid, params):
    rho = np.ones((grid.ny, grid.nx))
    fields = Fields(rho, np.full_like(rho, 0.2), np.zeros_like(rho))
    original = fields.copy()
    integrate(fields, grid, params, dt=0.001, n_steps=5)
    assert np.array_equal(fields.rho, original.rho)
    assert np.array_equal(fields.vx, original.vx)
    assert np.array_equal(fields.vy, original.vy)


def test_invalid_time_step(grid, params):
    fields = Fields(np.ones((grid.ny, grid.nx)), np.zeros((grid.ny, grid.nx)), np.zeros((grid.ny, grid.nx)))
    with pytest.raises(ValueError):
        step(fields, grid, params, dt=0.0)


def test_stability_bound_decreases_with_velocity_and_grid_refinement(params):
    coarse = Grid(nx=16, ny=16, lx=1.0, ly=1.0)
    fine = Grid(nx=32, ny=32, lx=1.0, ly=1.0)
    state_coarse = Fields(np.ones((16, 16)), np.ones((16, 16)), np.zeros((16, 16)))
    state_fine = Fields(np.ones((32, 32)), np.ones((32, 32)), np.zeros((32, 32)))
    dt_coarse = max_stable_dt(state_coarse, coarse, params)
    dt_fine = max_stable_dt(state_fine, fine, params)
    assert dt_fine < dt_coarse


def test_step_rejects_time_step_above_stability_limit(grid, params):
    fields = Fields(np.ones((grid.ny, grid.nx)), np.ones((grid.ny, grid.nx)), np.zeros((grid.ny, grid.nx)))
    stable_dt = max_stable_dt(fields, grid, params)
    with pytest.raises(ValueError, match="exceeds estimated stable step"):
        step(fields, grid, params, dt=2.0*stable_dt)


def test_integrate_automatically_substeps_large_requested_interval(grid, params):
    fields = Fields(np.ones((grid.ny, grid.nx)), np.full((grid.ny, grid.nx), 0.1), np.zeros((grid.ny, grid.nx)))
    result = integrate(fields, grid, params, dt=1.0, n_steps=1)
    assert np.isfinite(result.rho).all()
    assert np.isfinite(result.vx).all()
    assert np.isfinite(result.vy).all()
