
from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.boundary_conditions import (
    ChannelBoundaryConditions,
    apply_channel_boundary_conditions,
)
from toner_tu.channel_equations import compute_channel_rhs


def _combine(
    a: Fields,
    b: Fields,
    weight_a: float,
    weight_b: float,
) -> Fields:
    return Fields(
        rho=weight_a * a.rho + weight_b * b.rho,
        vx=weight_a * a.vx + weight_b * b.vx,
        vy=weight_a * a.vy + weight_b * b.vy,
    )


def step(
    fields: Fields,
    grid: Grid,
    params: TonerTuParameters,
    boundary: ChannelBoundaryConditions,
    dt: float,
) -> Fields:
    """Advance the channel fields by one SSP-RK2 time step."""

    if dt <= 0:
        raise ValueError("dt must be positive.")

    # Enforce boundary conditions on the initial state.
    state_0 = apply_channel_boundary_conditions(fields, boundary)

    # First Euler stage.
    rhs_0 = compute_channel_rhs(state_0, grid, params, boundary)

    stage_1 = Fields(
        rho=state_0.rho + dt * rhs_0.rho,
        vx=state_0.vx + dt * rhs_0.vx,
        vy=state_0.vy + dt * rhs_0.vy,
    )

    stage_1 = apply_channel_boundary_conditions(stage_1, boundary)

    # Second Euler stage.
    rhs_1 = compute_channel_rhs(stage_1, grid, params, boundary)

    euler_2 = Fields(
        rho=stage_1.rho + dt * rhs_1.rho,
        vx=stage_1.vx + dt * rhs_1.vx,
        vy=stage_1.vy + dt * rhs_1.vy,
    )

    # SSP-RK2 combination.
    result = _combine(
        state_0,
        euler_2,
        weight_a=0.5,
        weight_b=0.5,
    )

    return apply_channel_boundary_conditions(result, boundary)


def integrate(
    fields: Fields,
    grid: Grid,
    params: TonerTuParameters,
    boundary: ChannelBoundaryConditions,
    dt: float,
    n_steps: int,
) -> Fields:
    """Integrate the channel equations for n_steps."""

    if dt <= 0:
        raise ValueError("dt must be positive.")

    if not isinstance(n_steps, int) or n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer.")

    current = apply_channel_boundary_conditions(fields, boundary)

    for _ in range(n_steps):
        current = step(
            current,
            grid,
            params,
            boundary,
            dt,
        )

    return current
