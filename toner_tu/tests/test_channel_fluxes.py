
import numpy as np
import pytest

from toner_tu.fields import Grid
from toner_tu.channel_fluxes import (
    compute_upwind_fluxes,
    compute_density_rhs,
    density_cfl_rate,
)
from toner_tu.boundary_conditions import ChannelBoundaryConditions

@pytest.fixture
def boundary():
    return ChannelBoundaryConditions(
        rho_in=1.0,
        vx_in=1.0,
        vy_in=0.0,
    )

@pytest.fixture
def grid():
    return Grid(nx=8, ny=6, lx=8.0, ly=6.0)


def test_flux_shapes(grid, boundary):
    rho = np.ones((grid.ny, grid.nx))
    vx = np.ones_like(rho)
    vy = np.zeros_like(rho)

    flux_x, flux_y = compute_upwind_fluxes(
        rho, vx, vy, grid, boundary
    )

    assert flux_x.shape == (grid.ny, grid.nx + 1)
    assert flux_y.shape == (grid.ny + 1, grid.nx)


def test_impermeable_walls(grid, boundary):
    rho = np.ones((grid.ny, grid.nx))
    vx = np.zeros_like(rho)
    vy = np.ones_like(rho)

    flux_x, flux_y = compute_upwind_fluxes(
        rho, vx, vy, grid, boundary
    )

    assert np.allclose(flux_y[0, :], 0.0)
    assert np.allclose(flux_y[-1, :], 0.0)


def test_mass_balance(grid, boundary):
    rng = np.random.default_rng(42)

    rho = rng.uniform(0.1, 1.0, (grid.ny, grid.nx))
    vx = rng.uniform(-0.2, 0.2, rho.shape)
    vy = rng.uniform(-0.2, 0.2, rho.shape)

    rhs = compute_density_rhs(rho, vx, vy, grid, boundary)

    flux_x, flux_y = compute_upwind_fluxes(
        rho, vx, vy, grid, boundary
    )

    mass_rate_from_rhs = np.sum(rhs) * grid.dx * grid.dy

    boundary_mass_rate = -(
        np.sum(flux_x[:, -1]) - np.sum(flux_x[:, 0])
    ) * grid.dy

    assert mass_rate_from_rhs == pytest.approx(
        boundary_mass_rate
    )


def test_uniform_rightward_flow_preserves_mass(grid, boundary):
    rho = np.ones((grid.ny, grid.nx))
    vx = np.ones_like(rho)
    vy = np.zeros_like(rho)

    rhs = compute_density_rhs(
        rho, vx, vy, grid, boundary
    )

    assert np.allclose(rhs, 0.0)


def test_empty_region_stays_empty(grid, boundary):
    rho = np.zeros((grid.ny, grid.nx))
    rho[:, :2] = 1.0

    vx = np.ones_like(rho)
    vy = np.zeros_like(rho)

    rhs = compute_density_rhs(rho, vx, vy, grid, boundary)

    dt = 0.5
    updated_rho = rho + dt * rhs

    assert np.all(updated_rho >= -1e-12)


def test_cfl_rate_is_positive(grid, boundary):
    vx = np.ones((grid.ny, grid.nx))
    vy = np.zeros_like(vx)

    rate = density_cfl_rate(vx, vy, grid, boundary)

    assert rate > 0.0
    assert np.isfinite(rate)


def test_cfl_rate_uses_prescribed_left_boundary_velocity(grid):
    boundary = ChannelBoundaryConditions(
        rho_in=1.0,
        vx_in=-2.0,
        vy_in=0.0,
    )

    vx = np.zeros((grid.ny, grid.nx))
    vy = np.zeros_like(vx)

    rate = density_cfl_rate(vx, vy, grid, boundary)

    assert rate == pytest.approx(2.0 / grid.dx)


def test_positive_left_boundary_velocity_is_not_outflow(grid):
    boundary = ChannelBoundaryConditions(
        rho_in=1.0,
        vx_in=2.0,
        vy_in=0.0,
    )

    vx = np.zeros((grid.ny, grid.nx))
    vy = np.zeros_like(vx)

    rate = density_cfl_rate(vx, vy, grid, boundary)

    assert rate == pytest.approx(0.0)


def test_negative_density_is_rejected(grid):
    rho = np.ones((grid.ny, grid.nx))
    rho[2, 3] = -0.1

    vx = np.zeros_like(rho)
    vy = np.zeros_like(rho)

    with pytest.raises(ValueError):
        compute_density_rhs(rho, vx, vy, grid, boundary)


def test_density_rhs_is_conservative():
    grid = Grid(nx=20, ny=12, lx=10.0, ly=6.0)
    boundary = ChannelBoundaryConditions(
        rho_in=1.5,
        vx_in=0.8,
        vy_in=0.0,
    )

    rng = np.random.default_rng(42)

    rho = rng.uniform(0.5, 2.0, (grid.ny, grid.nx))
    vx = rng.uniform(0.2, 1.0, (grid.ny, grid.nx))
    vy = rng.uniform(-0.2, 0.2, (grid.ny, grid.nx))

    flux_x, flux_y = compute_upwind_fluxes(
        rho, vx, vy, grid, boundary
    )

    drho_dt = compute_density_rhs(rho, vx, vy, grid, boundary)

    mass_rate = np.sum(drho_dt) * grid.dx * grid.dy

    boundary_rate = -np.sum(
        flux_x[:, -1] - flux_x[:, 0]
    ) * grid.dy

    assert np.isclose(mass_rate, boundary_rate)


def test_left_boundary_uses_prescribed_velocity(grid):
    boundary = ChannelBoundaryConditions(
        rho_in=2.0,
        vx_in=0.7,
        vy_in=0.0,
    )

    rho = np.ones((grid.ny, grid.nx))
    vx = np.zeros_like(rho)
    vy = np.zeros_like(rho)

    flux_x, _ = compute_upwind_fluxes(
        rho, vx, vy, grid, boundary
    )

    assert np.allclose(flux_x[:, 0], 1.4)


def test_right_boundary_does_not_allow_external_inflow(grid):
    boundary = ChannelBoundaryConditions(
        rho_in=1.0,
        vx_in=0.0,
        vy_in=0.0,
    )

    rho = np.ones((grid.ny, grid.nx))
    vx = -np.ones_like(rho)
    vy = np.zeros_like(rho)

    flux_x, _ = compute_upwind_fluxes(
        rho, vx, vy, grid, boundary
    )

    assert np.allclose(flux_x[:, -1], 0.0)

def test_forward_euler_density_step_preserves_nonnegative_density(
    grid,
):
    boundary = ChannelBoundaryConditions(
        rho_in=1.0,
        vx_in=1.0,
        vy_in=0.0,
    )

    rng = np.random.default_rng(42)

    rho = rng.uniform(0.0, 1.0, size=(grid.ny, grid.nx))
    vx = rng.uniform(-2.0, 2.0, size=rho.shape)
    vy = rng.uniform(-2.0, 2.0, size=rho.shape)

    rate = density_cfl_rate(vx, vy, grid, boundary)
    dt = 0.9 / rate

    rhs = compute_density_rhs(rho, vx, vy, grid, boundary)
    rho_next = rho + dt * rhs

    assert np.all(rho_next >= -1e-14)