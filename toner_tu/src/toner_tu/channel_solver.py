
import numpy as np
from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.boundary_conditions import ChannelBoundaryConditions
from toner_tu.channel_equations import compute_channel_rhs
from toner_tu.channel_fluxes import density_cfl_rate


DENSITY_CFL_SAFETY = 0.9
REACTION_SAFETY = 0.9

class _TimeStepViolation(ValueError):
    """Raised when a time step violates a numerical stability bound."""

def _check_density_cfl(
    fields: Fields,
    grid: Grid,
    dt: float,
) -> None:
    rate = density_cfl_rate(
        fields.vx,
        fields.vy,
        grid,
    )

    if rate == 0.0:
        return

    max_dt = DENSITY_CFL_SAFETY / rate

    if dt > max_dt * (1.0 + 1e-12):
        raise _TimeStepViolation(
            f"Density CFL condition violated: "
            f"dt={dt:.6g}, maximum allowed dt={max_dt:.6g}."
        )

def _check_velocity_cfl(
    fields: Fields,
    grid: Grid,
    params: TonerTuParameters,
    dt: float,
) -> None:
    rate = velocity_cfl_rate(fields, grid, params)

    if rate == 0.0:
        return

    max_dt = DENSITY_CFL_SAFETY / rate

    if dt > max_dt * (1.0 + 1e-12):
        raise _TimeStepViolation(
            f"Velocity CFL condition violated: "
            f"dt={dt:.6g}, maximum allowed dt={max_dt:.6g}."
        )

def _check_max_stable_dt(fields, grid, params, dt):
    density_rate = density_cfl_rate(fields.vx, fields.vy, grid)

    if density_rate > 0.0:
        density_max_dt = DENSITY_CFL_SAFETY / density_rate

        if dt > density_max_dt:
            raise _TimeStepViolation(
                f"Density CFL condition violated: "
                f"dt={dt:.6g}, max_dt={density_max_dt:.6g}"
            )

    max_dt = max_stable_dt(fields, grid, params)

    if dt > max_dt:
        raise _TimeStepViolation(
            f"Time-step stability condition violated: "
            f"dt={dt:.6g}, max_dt={max_dt:.6g}"
        )

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

    state_0 = fields.copy()

    # First forward-Euler stage.
    _check_max_stable_dt(state_0, grid, params, dt)

    rhs_0 = compute_channel_rhs(
        state_0, grid, params, boundary
    )

    stage_1 = Fields(
        rho=state_0.rho + dt * rhs_0.rho,
        vx=state_0.vx + dt * rhs_0.vx,
        vy=state_0.vy + dt * rhs_0.vy,
    )

    # The velocity field has changed, so check CFL again.\
    _check_max_stable_dt(stage_1, grid, params, dt)

    rhs_1 = compute_channel_rhs(
        stage_1, grid, params, boundary
    )

    # Second forward-Euler stage.
    euler_2 = Fields(
        rho=stage_1.rho + dt * rhs_1.rho,
        vx=stage_1.vx + dt * rhs_1.vx,
        vy=stage_1.vy + dt * rhs_1.vy,
    )

    return _combine(
        state_0,
        euler_2,
        weight_a=0.5,
        weight_b=0.5,
    )


def integrate(
    fields: Fields,
    grid: Grid,
    params: TonerTuParameters,
    boundary: ChannelBoundaryConditions,
    dt: float,
    n_steps: int,
) -> Fields:
    """Integrate using adaptive substeps controlled by the density CFL bound."""

    if dt <= 0:
        raise ValueError("dt must be positive.")

    if not isinstance(n_steps, int) or n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer.")

    current = fields.copy()

    for _ in range(n_steps):
        elapsed = 0.0

        while elapsed < dt:
            remaining = dt - elapsed

            max_dt = max_stable_dt(current, grid, params)
            trial_dt = min(remaining, max_dt)

            # If the second RK stage violates CFL, retry with a
            # smaller step. This also handles a velocity increase
            # during the first stage.
            while True:
                try:
                    next_state = step(
                        current,
                        grid,
                        params,
                        boundary,
                        trial_dt,
                    )
                    break
                except _TimeStepViolation:
                    trial_dt *= 0.5

                    if trial_dt <= 1e-14 * max(1.0, dt):
                        raise RuntimeError(
                            "Could not find a sufficiently small "
                            "time step satisfying the density CFL condition."
                        )

            current = next_state
            elapsed += trial_dt

    return current

def reaction_rate(fields, params):
    speed_squared = fields.vx**2 + fields.vy**2

    lambda_perp = params.alpha - params.beta * speed_squared
    lambda_parallel = params.alpha - 3.0 * params.beta * speed_squared

    return float(
        np.max(
            np.maximum(
                np.abs(lambda_perp),
                np.abs(lambda_parallel),
            )
        )
    )

def reaction_max_dt(fields, params):
    rate = reaction_rate(fields, params)

    if rate == 0.0:
        return float("inf")

    return REACTION_SAFETY / rate

def velocity_cfl_rate(
    fields: Fields,
    grid: Grid,
    params: TonerTuParameters,
) -> float:
    """Estimate the explicit advection-diffusion stability rate."""

    advection_rate = abs(params.lambda_) * (
        np.max(np.abs(fields.vx)) / grid.dx
        + np.max(np.abs(fields.vy)) / grid.dy
    )

    diffusion_rate = 2.0 * params.diffusion * (
        1.0 / grid.dx**2
        + 1.0 / grid.dy**2
    )

    return float(advection_rate + diffusion_rate)

def max_stable_dt(fields, grid, params):
    density_rate = density_cfl_rate(
        fields.vx,
        fields.vy,
        grid,
    )
    velocity_rate = velocity_cfl_rate(
        fields,
        grid,
        params,
    )

    density_limit = (
        DENSITY_CFL_SAFETY / density_rate
        if density_rate > 0.0
        else float("inf")
    )

    velocity_limit = (
        DENSITY_CFL_SAFETY / velocity_rate
        if velocity_rate > 0.0
        else float("inf")
    )

    reaction_limit = reaction_max_dt(fields, params)

    return min(
        density_limit,
        velocity_limit,
        reaction_limit,
    )