
import numpy as np

from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.operators import (
    gradient,
    divergence,
    laplacian,
    advective_derivative,
)


def compute_rhs(
    fields: Fields,
    grid: Grid,
    params: TonerTuParameters,
) -> Fields:
    """
    Compute the right-hand side of the simplified 2D Toner-Tu equations.

    Periodic boundary conditions are assumed.
    """

    expected_shape = (grid.ny, grid.nx)

    if fields.rho.shape != expected_shape:
        raise ValueError(
            f"Expected field shape {expected_shape}, "
            f"got {fields.rho.shape}."
        )

    rho = fields.rho
    vx = fields.vx
    vy = fields.vy

    # Density equation:
    # d rho / dt = - div(rho * v)

    drho_dt = -divergence(rho * vx, rho * vy, grid)

    # Velocity equation:
    # dv/dt = -lambda * (v · grad)v
    #         + (alpha - beta * |v|^2) * v
    #         - sigma * grad(rho)
    #         + D * laplacian(v)

    adv_vx, adv_vy = advective_derivative(vx, vy, grid)

    grad_rho_x, grad_rho_y = gradient(rho, grid)

    lap_vx = laplacian(vx, grid)
    lap_vy = laplacian(vy, grid)

    speed_squared = vx**2 + vy**2

    local_growth = params.alpha - params.beta * speed_squared

    dvx_dt = (
        -params.lambda_ * adv_vx
        + local_growth * vx
        - params.sigma * grad_rho_x
        + params.diffusion * lap_vx
    )

    dvy_dt = (
        -params.lambda_ * adv_vy
        + local_growth * vy
        - params.sigma * grad_rho_y
        + params.diffusion * lap_vy
    )

    return Fields(
        rho=drho_dt,
        vx=dvx_dt,
        vy=dvy_dt,
    )
