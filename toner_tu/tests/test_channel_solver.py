
import numpy as np
import pytest

from toner_tu.channel_equations import compute_channel_rhs
from toner_tu.channel_fluxes import compute_upwind_fluxes, density_cfl_rate
from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.boundary_conditions import ChannelBoundaryConditions
from toner_tu.channel_solver import DENSITY_CFL_SAFETY, max_stable_dt, reaction_max_dt, step, integrate, velocity_cfl_rate
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

    with pytest.raises(ValueError, match="Time-step stability condition violated"):
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

def test_max_stable_dt_includes_reaction_limit(grid, boundary):
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

    result = channel_solver.max_stable_dt(fields, grid, params, boundary)

    assert result <= channel_solver.reaction_max_dt(fields, params)
    assert result > 0.0

@pytest.mark.parametrize("dt", [np.nan, np.inf, -np.inf])
def test_step_rejects_nonfinite_dt(grid, params, boundary, dt):
    shape = (grid.ny, grid.nx)

    fields = Fields(
        rho=np.ones(shape),
        vx=np.zeros(shape),
        vy=np.zeros(shape),
    )

    with pytest.raises(ValueError, match="finite and positive"):
        step(fields, grid, params, boundary, dt)


@pytest.mark.parametrize("dt", [np.nan, np.inf, -np.inf])
def test_integrate_rejects_nonfinite_dt(
    grid, params, boundary, dt
):
    shape = (grid.ny, grid.nx)

    fields = Fields(
        rho=np.ones(shape),
        vx=np.zeros(shape),
        vy=np.zeros(shape),
    )

    with pytest.raises(ValueError, match="finite and positive"):
        integrate(fields, grid, params, boundary, dt, 1)


def test_step_rejects_negative_initial_density(
    grid, params, boundary
):
    shape = (grid.ny, grid.nx)
    rho = np.ones(shape)
    rho[0, 0] = -0.1

    fields = Fields(
        rho=rho,
        vx=np.zeros(shape),
        vy=np.zeros(shape),
    )

    with pytest.raises(ValueError, match="Initial state"):
        step(fields, grid, params, boundary, 0.001)


def test_step_rejects_nonfinite_initial_velocity(
    grid, params, boundary
):
    shape = (grid.ny, grid.nx)
    vx = np.zeros(shape)
    vx[0, 0] = np.nan

    fields = Fields(
        rho=np.ones(shape),
        vx=vx,
        vy=np.zeros(shape),
    )

    with pytest.raises(ValueError, match="NaN or Inf"):
        step(fields, grid, params, boundary, 0.001)

@pytest.mark.parametrize("flow_sign", [1.0, -1.0])
def test_step_conserves_mass_according_to_boundary_fluxes(
    grid, flow_sign
):
    rng = np.random.default_rng(42)
    shape = (grid.ny, grid.nx)

    fields = Fields(
        rho=rng.uniform(0.5, 1.5, shape),
        vx=np.full(shape, 0.2 * flow_sign),
        vy=np.zeros(shape),
    )

    boundary = ChannelBoundaryConditions(
        rho_in=1.4,
        vx_in=0.2 * flow_sign,
        vy_in=0.0,
    )

    params = TonerTuParameters(
        alpha=0.0,
        beta=1.0,
        sigma=0.0,
        diffusion=0.0,
        lambda_=0.0,
    )

    dt = 0.01

    def boundary_mass_rate(state):
        flux_x, _ = compute_upwind_fluxes(
            state.rho,
            state.vx,
            state.vy,
            grid,
            boundary,
        )

        return -grid.dy * np.sum(
            flux_x[:, -1] - flux_x[:, 0]
        )

    rhs_0 = compute_channel_rhs(
        fields, grid, params, boundary
    )

    stage_1 = Fields(
        rho=fields.rho + dt * rhs_0.rho,
        vx=fields.vx + dt * rhs_0.vx,
        vy=fields.vy + dt * rhs_0.vy,
    )

    expected_mass_change = 0.5 * dt * (
        boundary_mass_rate(fields)
        + boundary_mass_rate(stage_1)
    )

    result = step(
        fields,
        grid,
        params,
        boundary,
        dt,
    )

    actual_mass_change = (
        np.sum(result.rho - fields.rho)
        * grid.dx
        * grid.dy
    )

    assert actual_mass_change == pytest.approx(
        expected_mass_change,
        rel=1e-10,
        abs=1e-12,
    )


@pytest.mark.parametrize(
    "vx_value, vx_in, rho_in, expected_sign",
    [
        (0.5, 0.5, 2.0, 1),    # Inflow on the left, outflow on the right.
        (-0.5, -0.5, 2.0, -1), # Outflow through the left boundary.
        (-0.5, 0.0, 2.0, 0),   # No external inflow through the right.
    ],
)
def test_step_respects_channel_boundary_mass_balance(
    grid, vx_value, vx_in, rho_in, expected_sign
):
    dt = 1e-3
    shape = (grid.ny, grid.nx)

    params = TonerTuParameters(
        alpha=0.25,
        beta=1.0,
        sigma=0.0,
        diffusion=0.0,
        lambda_=0.0,
    )
    boundary = ChannelBoundaryConditions(
        rho_in=rho_in,
        vx_in=vx_in,
        vy_in=0.0,
    )

    initial = Fields(
        rho=np.ones(shape),
        vx=np.full(shape, vx_value),
        vy=np.zeros(shape),
    )

    def boundary_mass_rate(state):
        flux_x, _ = compute_upwind_fluxes(
            state.rho,
            state.vx,
            state.vy,
            grid,
            boundary,
        )
        return grid.dy * np.sum(flux_x[:, 0] - flux_x[:, -1])

    # Reconstruct the first SSP-RK2 stage to calculate its boundary flux.
    rhs_0 = compute_channel_rhs(initial, grid, params, boundary)
    stage_1 = Fields(
        rho=initial.rho + dt * rhs_0.rho,
        vx=initial.vx + dt * rhs_0.vx,
        vy=initial.vy + dt * rhs_0.vy,
    )

    expected_mass_change = 0.5 * dt * (
        boundary_mass_rate(initial) + boundary_mass_rate(stage_1)
    )

    result = step(initial, grid, params, boundary, dt)

    actual_mass_change = (
        np.sum(result.rho - initial.rho) * grid.dx * grid.dy
    )

    assert actual_mass_change == pytest.approx(
        expected_mass_change, abs=1e-11
    )

    if expected_sign > 0:
        assert actual_mass_change > 0.0
    elif expected_sign < 0:
        assert actual_mass_change < 0.0
    else:
        assert actual_mass_change == pytest.approx(0.0, abs=1e-11)


def test_max_stable_dt_is_minimum_of_all_stability_limits():
    grid = Grid(nx=16, ny=16, lx=10.0, ly=10.0)
    fields = Fields(
        rho=np.ones((grid.ny, grid.nx)),
        vx=np.full((grid.ny, grid.nx), 0.5),
        vy=np.full((grid.ny, grid.nx), 0.2),
    )
    params = TonerTuParameters(
        alpha=0.5,
        beta=1.0,
        sigma=0.0,
        diffusion=0.1,
        lambda_=1.0,
    )
    boundary = ChannelBoundaryConditions(
        rho_in=1.0,
        vx_in=0.5,
        vy_in=0.0,
    )

    density_rate = density_cfl_rate(
        fields.vx, fields.vy, grid, boundary
    )
    velocity_rate = velocity_cfl_rate(fields, grid, params)

    density_limit = (
        DENSITY_CFL_SAFETY / density_rate
        if density_rate > 0.0
        else float("inf")
    )
    velocity_limit = (
        DENSITY_CFL_SAFETY / velocity_rate
        if velocity_rate > 0.0
        else float("inf")
    )
    reaction_limit = reaction_max_dt(fields, params)

    expected = min(
        density_limit,
        velocity_limit,
        reaction_limit,
    )

    actual = max_stable_dt(fields, grid, params, boundary)

    assert np.isclose(actual, expected)