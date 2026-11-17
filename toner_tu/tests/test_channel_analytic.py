
import numpy as np

from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.boundary_conditions import ChannelBoundaryConditions
from toner_tu.channel_solver import integrate


def test_uniform_velocity_relaxation_matches_analytic_solution():
    grid = Grid(nx=16, ny=8, lx=20.0, ly=10.0)

    alpha = 1.0
    beta = 1.0
    v0 = 0.2
    final_time = 0.1

    params = TonerTuParameters(
        alpha=alpha,
        beta=beta,
        sigma=0.0,
        diffusion=0.0,
        lambda_=0.0,
    )

    boundary = ChannelBoundaryConditions(
        rho_in=1.0,
        vx_in=v0,
        vy_in=0.0,
    )

    initial = Fields(
        rho=np.ones((grid.ny, grid.nx)),
        vx=np.full((grid.ny, grid.nx), v0),
        vy=np.zeros((grid.ny, grid.nx)),
    )

    numerical = integrate(
        initial,
        grid,
        params,
        boundary,
        dt=0.001,
        n_steps=100,
    )

    # Exact solution of dv/dt = (alpha - beta*v^2)*v
    coefficient = alpha / (beta * v0**2) - 1.0

    exact_velocity = np.sqrt(
        (alpha / beta)
        / (1.0 + coefficient * np.exp(-2.0 * alpha * final_time))
    )

    np.testing.assert_allclose(
        numerical.vx,
        exact_velocity,
        rtol=2e-5,
        atol=1e-8,
    )

    np.testing.assert_allclose(
        numerical.vy,
        0.0,
        atol=1e-10,
    )

    assert np.all(np.isfinite(numerical.rho))
    assert np.all(numerical.rho >= 0.0)