
import numpy as np
import pytest

from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.boundary_conditions import ChannelBoundaryConditions
from toner_tu.channel_solver import step, integrate
from toner_tu import channel_solver


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



def test_step_rejects_dt_violating_density_cfl(grid, params, boundary):
    shape = (grid.ny, grid.nx)

    fields = Fields(
        rho=np.ones(shape),
        vx=np.full(shape, 10.0),
        vy=np.zeros(shape),
    )

    with pytest.raises(ValueError, match="Density CFL condition violated"):
        step(
            fields,
            grid,
            params,
            boundary,
            dt=0.1,
        )


def test_integrate_automatically_substeps_density_cfl(
    grid, params, boundary, monkeypatch
):
    shape = (grid.ny, grid.nx)

    fields = Fields(
        rho=np.ones(shape),
        vx=np.full(shape, 10.0),
        vy=np.zeros(shape),
    )

    calls = []

    zero_rhs = Fields(
        rho=np.zeros(shape),
        vx=np.zeros(shape),
        vy=np.zeros(shape),
    )

    def mock_rhs(*args):
        calls.append(1)
        return zero_rhs

    monkeypatch.setattr(
        channel_solver,
        "compute_channel_rhs",
        mock_rhs,
    )

    result = integrate(
        fields,
        grid,
        params,
        boundary,
        dt=0.1,
        n_steps=1,
    )

    # At least two RK2 steps were performed.
    assert len(calls) >= 4

    assert np.allclose(result.rho, fields.rho)
    assert np.allclose(result.vx, fields.vx)
    assert np.allclose(result.vy, fields.vy)

def test_reaction_rate():
    fields = Fields(
        rho=np.ones((4, 4)),
        vx=np.full((4, 4), 0.5),
        vy=np.full((4, 4), 0.0),
    )

    params = TonerTuParameters(
        alpha=1.0,
        beta=1.0,
        sigma=1.0,
        diffusion=0.1,
        lambda_=1.0,
    )

    rate = channel_solver.reaction_rate(fields, params)

    assert rate == pytest.approx(0.75)


def test_reaction_max_dt():
    fields = Fields(
        rho=np.ones((4, 4)),
        vx=np.full((4, 4), 0.5),
        vy=np.zeros((4, 4)),
    )

    params = TonerTuParameters(
        alpha=1.0,
        beta=1.0,
        sigma=1.0,
        diffusion=0.1,
        lambda_=1.0,
    )

    max_dt = channel_solver.reaction_max_dt(fields, params)

    assert max_dt == pytest.approx(0.9 / 0.75)

def test_reaction_max_dt_is_infinite_when_rate_is_zero():
    fields = Fields(
        rho=np.ones((4, 4)),
        vx=np.zeros((4, 4)),
        vy=np.zeros((4, 4)),
    )

    params = TonerTuParameters(
        alpha=0.0,
        beta=1.0,
        sigma=1.0,
        diffusion=0.1,
        lambda_=1.0,
    )

    assert channel_solver.reaction_rate(fields, params) == pytest.approx(0.0)
    assert channel_solver.reaction_max_dt(fields, params) == float("inf")


def test_reaction_rate_includes_radial_eigenvalue():
    fields = Fields(
        rho=np.ones((4, 4)),
        vx=np.ones((4, 4)),
        vy=np.zeros((4, 4)),
    )

    params = TonerTuParameters(
        alpha=1.0,
        beta=1.0,
        sigma=1.0,
        diffusion=0.1,
        lambda_=1.0,
    )

    assert channel_solver.reaction_rate(fields, params) == pytest.approx(2.0)

def test_max_stable_dt_includes_reaction_limit(grid):
    fields = Fields(
        rho=np.ones((grid.ny, grid.nx)),
        vx=np.full((grid.ny, grid.nx), 0.5),
        vy=np.zeros((grid.ny, grid.nx)),
    )

    params = TonerTuParameters(
        alpha=1.0,
        beta=1.0,
        sigma=1.0,
        diffusion=0.1,
        lambda_=1.0,
    )

    result = channel_solver.max_stable_dt(fields, grid, params)

    assert result <= channel_solver.reaction_max_dt(fields, params)
    assert result > 0.0