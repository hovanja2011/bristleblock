
from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters

from toner_tu.channel_operators import (
    gradient,
    divergence,
    laplacian,
    advective_derivative,
)

from toner_tu.boundary_conditions import (
    ChannelBoundaryConditions,
    apply_channel_boundary_conditions,
)


def compute_channel_rhs(
    fields: Fields,
    grid: Grid,
    params: TonerTuParameters,
    boundary: ChannelBoundaryConditions,
) -> Fields:
    """
    Compute the Toner-Tu right-hand side in an open channel.

    Boundary conditions are applied before calculating derivatives.
    """

    expected_shape = (grid.ny, grid.nx)

    if fields.rho.shape != expected_shape:
        raise ValueError(
            f"Expected field shape {expected_shape}, "
            f"got {fields.rho.shape}."
        )

    # Work with a state satisfying the prescribed boundary values.
    state = apply_channel_boundary_conditions(fields, boundary)

    rho = state.rho
    vx = state.vx
    vy = state.vy

    # Density equation:
    # d rho / dt = - div(rho * v)

    drho_dt = -divergence(
        rho * vx,
        rho * vy,
        grid,
    )

    # Velocity equation:
    # dv/dt = -lambda (v · grad)v
    #         + (alpha - beta |v|²)v
    #         - sigma grad(rho)
    #         + D laplacian(v)

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
