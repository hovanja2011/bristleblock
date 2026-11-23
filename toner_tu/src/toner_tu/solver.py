"""SSP-RK2 integration and domain-independent explicit stability control."""
import numpy as np

from toner_tu.domain import DomainOperators
from toner_tu.equations import compute_rhs
from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters


def max_stable_dt(fields: Fields, grid: Grid, params: TonerTuParameters,
                  cfl: float = 0.4) -> float:
    """Conservative explicit time-step bound for the Toner–Tu PDE.

    Includes advection, density/velocity compressional coupling, diffusion,
    and local ordering reaction. It depends on fields, coefficients and grid,
    not on a particular maze geometry.
    """
    if not np.isfinite(cfl) or not 0.0 < cfl <= 1.0:
        raise ValueError("cfl must be finite and in (0, 1].")
    expected = (grid.ny, grid.nx)
    arrays = (fields.rho, fields.vx, fields.vy)
    if any(a.shape != expected for a in arrays):
        raise ValueError(f"All fields must have shape {expected}.")
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Fields must contain only finite values.")
    if np.any(fields.rho < 0):
        raise ValueError("Density must be non-negative.")

    # Linearized density-pressure coupling propagates at sqrt(sigma*rho).
    wave_speed = float(np.sqrt(params.sigma * np.max(fields.rho)))
    speed_x = float(np.max(np.abs(fields.vx))) * max(1.0, abs(params.lambda_))
    speed_y = float(np.max(np.abs(fields.vy))) * max(1.0, abs(params.lambda_))
    advective_rate = (speed_x + wave_speed) / grid.dx + (speed_y + wave_speed) / grid.dy

    # Conservative bound for the five-point explicit diffusion operator.
    diffusion_rate = 2.0 * params.diffusion * (grid.dx**-2 + grid.dy**-2)

    # Bound the Jacobian of the local cubic ordering term.
    speed_sq = fields.vx**2 + fields.vy**2
    reaction_rate = float(np.max(abs(params.alpha) + 3.0 * params.beta * speed_sq))

    total_rate = advective_rate + diffusion_rate + reaction_rate
    return float("inf") if total_rate == 0.0 else cfl / total_rate


def _combine(a: Fields, b: Fields, wa: float, wb: float) -> Fields:
    """Return the weighted sum wa*a + wb*b for all three state fields."""
    return Fields(wa*a.rho + wb*b.rho, wa*a.vx + wb*b.vx, wa*a.vy + wb*b.vy)


def step(fields: Fields, grid: Grid, params: TonerTuParameters, dt: float,
         domain: DomainOperators | None = None, *, check_stability: bool = True) -> Fields:
    """Advance one SSP-RK2 step, rejecting a step above the estimated limit."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Time step dt must be finite and positive.")
    if check_stability:
        stable_dt = max_stable_dt(fields, grid, params)
        if dt > stable_dt * (1.0 + 1e-12):
            raise ValueError(
                f"dt={dt:g} exceeds estimated stable step {stable_dt:g}; "
                "reduce dt or use integrate() for automatic substepping."
            )

    # SSP-RK2: Euler stage, second Euler stage, then a convex blend with u^n.
    rhs0 = compute_rhs(fields, grid, params, domain)
    stage1 = Fields(fields.rho + dt*rhs0.rho, fields.vx + dt*rhs0.vx, fields.vy + dt*rhs0.vy)
    if np.any(stage1.rho < 0) or not all(np.isfinite(a).all() for a in (stage1.rho, stage1.vx, stage1.vy)):
        raise ValueError("First RK stage produced an invalid state.")
    rhs1 = compute_rhs(stage1, grid, params, domain)
    euler2 = Fields(stage1.rho + dt*rhs1.rho, stage1.vx + dt*rhs1.vx, stage1.vy + dt*rhs1.vy)
    result = _combine(fields, euler2, 0.5, 0.5)
    if np.any(result.rho < 0) or not all(np.isfinite(a).all() for a in (result.rho, result.vx, result.vy)):
        raise ValueError("RK2 step produced an invalid state.")
    return result


def integrate(fields: Fields, grid: Grid, params: TonerTuParameters, dt: float,
              n_steps: int, domain: DomainOperators | None = None, *,
              cfl: float = 0.4) -> Fields:
    """Integrate n output intervals with state-dependent adaptive substepping.

    dt is the requested physical interval per outer step. The stable bound is
    recomputed after every substep, so growth of speed or density cannot leave
    the remaining part of an interval using a stale estimate.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Time step dt must be finite and positive.")
    if not isinstance(n_steps, int) or n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer.")
    if not np.isfinite(cfl) or not 0.0 < cfl <= 1.0:
        raise ValueError("cfl must be finite and in (0, 1].")

    current = fields.copy()
    for _ in range(n_steps):
        remaining = dt
        while remaining > 1e-14 * dt:
            stable_dt = max_stable_dt(current, grid, params, cfl=cfl)
            sub_dt = remaining if np.isinf(stable_dt) else min(remaining, stable_dt)
            current = step(current, grid, params, sub_dt, domain, check_stability=False)
            remaining -= sub_dt
    return current
