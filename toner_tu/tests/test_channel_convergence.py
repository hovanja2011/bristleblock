import numpy as np

from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.boundary_conditions import ChannelBoundaryConditions
from toner_tu.channel_solver import integrate


def test_channel_solver_has_second_order_temporal_convergence():
    grid = Grid(nx=64, ny=32, lx=20.0, ly=10.0)

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

    x, y = grid.coordinates()

    rho = 1.0 + 0.05 * np.sin(2.0 * np.pi * x / grid.lx)
    vx = 0.5 + 0.02 * np.sin(2.0 * np.pi * x / grid.lx)
    vy = 0.01 * np.sin(2.0 * np.pi * y / grid.ly)

    initial = Fields(rho=rho, vx=vx, vy=vy)

    final_time = 0.01
    dt = 0.0005

    solution_dt = integrate(
        initial, grid, params, boundary,
        dt, round(final_time / dt),
    )

    solution_half = integrate(
        initial, grid, params, boundary,
        dt / 2, round(final_time / (dt / 2)),
    )

    solution_quarter = integrate(
        initial, grid, params, boundary,
        dt / 4, round(final_time / (dt / 4)),
    )

    def error(a, b):
        return np.sqrt(
            np.mean((a.rho - b.rho) ** 2)
            + np.mean((a.vx - b.vx) ** 2)
            + np.mean((a.vy - b.vy) ** 2)
        )

    error_coarse = error(solution_dt, solution_half)
    error_fine = error(solution_half, solution_quarter)

    ratio = error_coarse / error_fine

    assert ratio > 3.0
    assert ratio < 5.0