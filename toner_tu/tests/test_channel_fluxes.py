
import numpy as np
import pytest

from toner_tu.fields import Grid
from toner_tu.channel_fluxes import (
    compute_upwind_fluxes,
    compute_density_rhs,
    density_cfl_rate,
)


@pytest.fixture
def grid():
    return Grid(nx=8, ny=6, lx=8.0, ly=6.0)


def test_flux_shapes(grid):
    rho = np.ones((grid.ny, grid.nx))
    vx = np.ones_like(rho)
    vy = np.zeros_like(rho)

    flux_x, flux_y = compute_upwind_fluxes(
        rho, vx, vy, grid
    )

    assert flux_x.shape == (grid.ny, grid.nx + 1)
    assert flux_y.shape == (grid.ny + 1, grid.nx)


def test_impermeable_walls(grid):
    rho = np.ones((grid.ny, grid.nx))
    vx = np.zeros_like(rho)
    vy = np.ones_like(rho)

    flux_x, flux_y = compute_upwind_fluxes(
        rho, vx, vy, grid
    )

    assert np.allclose(flux_y[0, :], 0.0)
    assert np.allclose(flux_y[-1, :], 0.0)


def test_mass_balance(grid):
    rng = np.random.default_rng(42)

    rho = rng.uniform(0.1, 1.0, (grid.ny, grid.nx))
    vx = rng.uniform(-0.2, 0.2, rho.shape)
    vy = rng.uniform(-0.2, 0.2, rho.shape)

    rhs = compute_density_rhs(rho, vx, vy, grid)

    flux_x, flux_y = compute_upwind_fluxes(
        rho, vx, vy, grid
    )

    mass_rate_from_rhs = np.sum(rhs) * grid.dx * grid.dy

    boundary_mass_rate = -(
        np.sum(flux_x[:, -1]) - np.sum(flux_x[:, 0])
    ) * grid.dy

    assert mass_rate_from_rhs == pytest.approx(
        boundary_mass_rate
    )


def test_uniform_rightward_flow_loses_mass(grid):
    rho = np.ones((grid.ny, grid.nx))
    vx = np.ones_like(rho)
    vy = np.zeros_like(rho)

    rhs = compute_density_rhs(rho, vx, vy, grid)

    # No inflow from the left, outflow at the right.
    assert np.allclose(rhs[:, 0], -1.0 / grid.dx)
    assert np.allclose(rhs[:, 1:-1], 0.0)
    assert np.allclose(rhs[:, -1], 0.0)

    assert np.sum(rhs) < 0.0


def test_empty_region_stays_empty(grid):
    rho = np.zeros((grid.ny, grid.nx))
    rho[:, :2] = 1.0

    vx = np.ones_like(rho)
    vy = np.zeros_like(rho)

    rhs = compute_density_rhs(rho, vx, vy, grid)

    dt = 0.5
    updated_rho = rho + dt * rhs

    assert np.all(updated_rho >= -1e-12)


def test_cfl_rate_is_positive(grid):
    rho = np.ones((grid.ny, grid.nx))
    vx = np.ones_like(rho)
    vy = np.zeros_like(rho)

    rate = density_cfl_rate(vx, vy, grid)

    assert rate > 0.0
    assert np.isfinite(rate)


def test_negative_density_is_rejected(grid):
    rho = np.ones((grid.ny, grid.nx))
    rho[2, 3] = -0.1

    vx = np.zeros_like(rho)
    vy = np.zeros_like(rho)

    with pytest.raises(ValueError):
        compute_density_rhs(rho, vx, vy, grid)
