from toner_tu.channel_fluxes import compute_density_rhs
from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.boundary_conditions import ChannelBoundaryConditions, add_ghost_cells

from toner_tu.channel_operators import (
    gradient,
    laplacian,
    advective_derivative_upwind,
)


def compute_channel_rhs(
    fields: Fields,
    grid: Grid,
    params: TonerTuParameters,
    boundary: ChannelBoundaryConditions,
) -> Fields:
    """Compute the Toner-Tu right-hand side in an open channel."""

    expected_shape = (grid.ny, grid.nx)

    if fields.rho.shape != expected_shape:
        raise ValueError(
            f"Expected field shape {expected_shape}, "
            f"got {fields.rho.shape}."
        )

    rho = fields.rho
    vx = fields.vx
    vy = fields.vy

    # Conservative density transport.
    drho_dt = compute_density_rhs(
        rho, vx, vy, grid, boundary
    )

    vx_extended = add_ghost_cells(vx, boundary, "vx")
    vy_extended = add_ghost_cells(vy, boundary, "vy")

    # Advective derivative.
    adv_vx, adv_vy = advective_derivative_upwind(
        vx,
        vy,
        grid,
        boundary,
        vx_extended=vx_extended,
        vy_extended=vy_extended,
    )

    # Density gradient.
    grad_rho_x, grad_rho_y = gradient(
        rho, grid, boundary, "rho"
    )

    # Velocity Laplacians.
    lap_vx = laplacian(vx, grid, boundary, "vx", extended=vx_extended)
    lap_vy = laplacian(vy, grid, boundary, "vy", extended=vy_extended)

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