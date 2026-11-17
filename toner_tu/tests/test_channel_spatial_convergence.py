import numpy as np

from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.boundary_conditions import ChannelBoundaryConditions
from toner_tu.channel_solver import integrate


def test_channel_solver_spatial_convergence():
    params = TonerTuParameters(
        alpha=1.0,
        beta=1.0,
        sigma=0.1,
        diffusion=0.05,
        lambda_=0.5,
    )

    boundary = ChannelBoundaryConditions(
        rho_in=1.0,
        vx_in=0.5,
        vy_in=0.0,
    )

    final_time = 0.005
    dt = 0.0001

    solutions = []

    for nx, ny in [(32, 16), (64, 32), (128, 64)]:
        grid = Grid(
            nx=nx,
            ny=ny,
            lx=20.0,
            ly=10.0,
        )

        x, y = grid.coordinates()

        rho = 1.0 + 0.05 * np.sin(2.0 * np.pi * x / grid.lx)
        vx = 0.5 + 0.02 * np.sin(2.0 * np.pi * x / grid.lx)
        vy = 0.01 * np.sin(2.0 * np.pi * y / grid.ly)

        initial = Fields(rho=rho, vx=vx, vy=vy)

        result = integrate(
            initial,
            grid,
            params,
            boundary,
            dt,
            round(final_time / dt),
        )

        solutions.append((grid, result))

    coarse_grid, coarse = solutions[0]
    medium_grid, medium = solutions[1]
    fine_grid, fine = solutions[2]

    def restrict(field):
        return field[::2, ::2]

    def error(a, b):
        return np.sqrt(np.mean((a - b) ** 2))

    rho_error_coarse = error(
        coarse.rho,
        restrict(medium.rho),
    )

    rho_error_fine = error(
        medium.rho,
        restrict(fine.rho),
    )

    assert rho_error_coarse > rho_error_fine