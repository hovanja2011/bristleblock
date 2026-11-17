
import numpy as np

from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.equations import compute_rhs


def _combine(a: Fields, b: Fields, weight_a: float, weight_b: float) -> Fields:
    """Return weight_a * a + weight_b * b."""
    return Fields(
        rho=weight_a * a.rho + weight_b * b.rho,
        vx=weight_a * a.vx + weight_b * b.vx,
        vy=weight_a * a.vy + weight_b * b.vy,
    )


def step(
    fields: Fields,
    grid: Grid,
    params: TonerTuParameters,
    dt: float,
) -> Fields:
    """
    Advance the Toner-Tu fields by one SSP-RK2 time step.
    """

    if dt <= 0:
        raise ValueError("Time step dt must be positive.")

    # First stage: Euler step
    rhs_0 = compute_rhs(fields, grid, params)

    stage_1 = Fields(
        rho=fields.rho + dt * rhs_0.rho,
        vx=fields.vx + dt * rhs_0.vx,
        vy=fields.vy + dt * rhs_0.vy,
    )

    # Second stage
    rhs_1 = compute_rhs(stage_1, grid, params)

    result = _combine(
        Fields(
            rho=stage_1.rho + dt * rhs_1.rho,
            vx=stage_1.vx + dt * rhs_1.vx,
            vy=stage_1.vy + dt * rhs_1.vy,
        ),
        fields,
        0.5,
        0.5,
    )

    return result


def integrate(
    fields: Fields,
    grid: Grid,
    params: TonerTuParameters,
    dt: float,
    n_steps: int,
) -> Fields:
    """Advance the fields through n_steps time steps."""

    if dt <= 0:
        raise ValueError("Time step dt must be positive.")

    if not isinstance(n_steps, int) or n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer.")

    current = fields.copy()

    for _ in range(n_steps):
        current = step(current, grid, params, dt)

    return current
