"""Domain-independent Toner–Tu equations for a 2D polar active fluid."""
import numpy as np
from toner_tu.domain import DomainOperators, PeriodicDomain
from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters

def compute_rhs(fields: Fields, grid: Grid, params: TonerTuParameters, domain: DomainOperators | None = None) -> Fields:
    """Evaluate the physical RHS; domain supplies spatial and boundary operators."""
    expected = (grid.ny, grid.nx)
    for name, value in (("rho", fields.rho), ("vx", fields.vx), ("vy", fields.vy)):
        if value.shape != expected: raise ValueError(f"{name}: expected shape {expected}, got {value.shape}")
        if not np.isfinite(value).all(): raise ValueError(f"{name} contains NaN or Inf")
    if np.any(fields.rho < 0): raise ValueError("Density must be non-negative")
    # Periodic operators are the reference domain; other boundaries live in mazes/.
    ops = domain if domain is not None else PeriodicDomain()
    rho, vx, vy = fields.rho, fields.vx, fields.vy
    # Conservative continuity equation: flux discretization belongs to the domain.
    drho = ops.density_rhs(rho, vx, vy, grid)
    # Universal constitutive Toner–Tu velocity terms.
    adv_x, adv_y = ops.advective_derivative(vx, vy, grid)
    grad_x, grad_y = ops.gradient(rho, grid)
    lap_x, lap_y = ops.laplacian(vx, grid, "vx"), ops.laplacian(vy, grid, "vy")
    growth = params.alpha - params.beta * (vx**2 + vy**2)
    dvx = -params.lambda_*adv_x + growth*vx - params.sigma*grad_x + params.diffusion*lap_x
    dvy = -params.lambda_*adv_y + growth*vy - params.sigma*grad_y + params.diffusion*lap_y
    return Fields(drho, dvx, dvy)