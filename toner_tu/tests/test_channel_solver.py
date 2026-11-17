
import numpy as np
import pytest

from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.boundary_conditions import ChannelBoundaryConditions
from toner_tu.channel_solver import step, integrate


@pytest.fixture
def grid():
    return Grid(nx=16, ny=16, lx=10.0, ly=10.0)


@pytest.fixture
def params():
    return TonerTuParameters(
        alpha=1.0,
        beta=1.0,
        sigma=1.0,
        diffusion=0.1,
        lambda_=1.0,
    )


@pytest.fixture
def boundary():
    return ChannelBoundaryConditions(
        rho_in=1.0,
        vx_in=1.0,
        vy_in=0.0,
    )


def test_step_preserves_uniform_equilibrium(grid, params, boundary):
    shape = (grid.ny, grid.nx)

    fields = Fields(
        rho=np.ones(shape),
        vx=np.ones(shape),
        vy=np.zeros(shape),
    )

    result = step(
        fields,
        grid,
        params,
        boundary,
        dt=0.001,
    )

    assert np.allclose(result.rho, 1.0)
    assert np.allclose(result.vx, 1.0)
    assert np.allclose(result.vy, 0.0)


def test_integrate_zero_steps_returns_copy(grid, params, boundary):
    shape = (grid.ny, grid.nx)

    fields = Fields(
        rho=np.ones(shape),
        vx=np.ones(shape),
        vy=np.zeros(shape),
    )

    result = integrate(
        fields,
        grid,
        params,
        boundary,
        dt=0.001,
        n_steps=0,
    )

    assert result is not fields
    assert np.allclose(result.rho, fields.rho)
    assert np.allclose(result.vx, fields.vx)
    assert np.allclose(result.vy, fields.vy)


def test_integrate_runs_multiple_steps(grid, params, boundary):
    shape = (grid.ny, grid.nx)

    fields = Fields(
        rho=np.ones(shape),
        vx=np.ones(shape),
        vy=np.zeros(shape),
    )

    result = integrate(
        fields,
        grid,
        params,
        boundary,
        dt=0.001,
        n_steps=10,
    )

    assert result.rho.shape == shape
    assert np.all(np.isfinite(result.rho))
    assert np.all(np.isfinite(result.vx))
    assert np.all(np.isfinite(result.vy))


def test_invalid_dt(grid, params, boundary):
    fields = Fields(
        rho=np.ones((grid.ny, grid.nx)),
        vx=np.ones((grid.ny, grid.nx)),
        vy=np.zeros((grid.ny, grid.nx)),
    )

    with pytest.raises(ValueError):
        step(fields, grid, params, boundary, dt=0.0)
