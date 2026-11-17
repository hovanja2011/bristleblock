import numpy as np

from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.boundary_conditions import ChannelBoundaryConditions
from toner_tu.channel_fluxes import compute_upwind_fluxes
from toner_tu.channel_fluxes import compute_density_rhs
from toner_tu.channel_solver import integrate

def test_density_rhs_is_conservative():
    grid = Grid(nx=32, ny=16, lx=20.0, ly=10.0)

    boundary = ChannelBoundaryConditions(
        rho_in=1.0,
        vx_in=0.5,
        vy_in=0.0,
    )

    x, y = grid.coordinates()

    rho = 1.0 + 0.1 * np.sin(2.0 * np.pi * x / grid.lx)
    vx = np.full_like(rho, 0.5)
    vy = np.zeros_like(rho)

    flux_x, flux_y = compute_upwind_fluxes(
        rho, vx, vy, grid, boundary
    )

    rhs = -(
        (flux_x[:, 1:] - flux_x[:, :-1]) / grid.dx
        + (flux_y[1:, :] - flux_y[:-1, :]) / grid.dy
    )

    mass_rate = np.sum(rhs) * grid.dx * grid.dy

    boundary_rate = (
        np.sum(flux_x[:, 0]) - np.sum(flux_x[:, -1])
    ) * grid.dy

    assert np.isclose(mass_rate, boundary_rate)


def test_density_remains_nonnegative():
    grid = Grid(nx=64, ny=16, lx=20.0, ly=5.0)

    boundary = ChannelBoundaryConditions(
        rho_in=0.1,
        vx_in=1.0,
        vy_in=0.0,
    )

    rho = np.ones((grid.ny, grid.nx))
    rho[:, 0] = 0.01
    rho[:, 1:4] = 0.05

    vx = np.ones_like(rho)
    vy = np.zeros_like(rho)

    fields = Fields(rho=rho, vx=vx, vy=vy)

    rhs = compute_density_rhs(
        rho, vx, vy, grid, boundary
    )

    dt = 0.5 * grid.dx

    rho_next = rho + dt * rhs

    assert np.min(rho_next) >= -1e-12

def test_integrated_density_mass_balance():
    grid = Grid(nx=32, ny=16, lx=20.0, ly=10.0)

    params = TonerTuParameters(
        alpha=0.25,
        beta=1.0,
        sigma=0.0,
        diffusion=0.0,
        lambda_=0.0,
    )

    boundary = ChannelBoundaryConditions(
        rho_in=1.0,
        vx_in=0.5,
        vy_in=0.0,
    )

    rho = np.ones((grid.ny, grid.nx))
    vx = np.full_like(rho, 0.5)
    vy = np.zeros_like(rho)

    initial = Fields(rho=rho, vx=vx, vy=vy)

    initial_mass = np.sum(rho) * grid.dx * grid.dy

    dt = 0.001
    n_steps = 10

    result = integrate(
        initial,
        grid,
        params,
        boundary,
        dt,
        n_steps,
    )

    final_mass = np.sum(result.rho) * grid.dx * grid.dy

    assert np.isclose(
        final_mass,
        initial_mass,
        rtol=1e-10,
        atol=1e-10,
    )

    assert np.allclose(result.rho, 1.0, atol=1e-10)
    assert np.allclose(result.vx, 0.5, atol=1e-10)
    assert np.allclose(result.vy, 0.0, atol=1e-10)