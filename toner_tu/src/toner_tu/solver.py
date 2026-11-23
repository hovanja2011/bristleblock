"""SSP-RK2 time integration for the domain-independent Toner–Tu model."""
import numpy as np
from toner_tu.domain import DomainOperators
from toner_tu.equations import compute_rhs
from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters

def _combine(a, b, wa, wb):
    """Combine two complete hydrodynamic states."""
    return Fields(wa*a.rho+wb*b.rho, wa*a.vx+wb*b.vx, wa*a.vy+wb*b.vy)

def step(fields: Fields, grid: Grid, params: TonerTuParameters, dt: float, domain: DomainOperators | None = None) -> Fields:
    """Advance one second-order SSP-RK2 step with the supplied domain operators."""
    if not np.isfinite(dt) or dt <= 0: raise ValueError("Time step dt must be finite and positive.")
    # SSP-RK2: two forward-Euler stages followed by a convex blend.
    rhs0 = compute_rhs(fields, grid, params, domain)
    stage1 = Fields(fields.rho+dt*rhs0.rho, fields.vx+dt*rhs0.vx, fields.vy+dt*rhs0.vy)
    if np.any(stage1.rho < 0) or not all(np.isfinite(a).all() for a in (stage1.rho,stage1.vx,stage1.vy)): raise ValueError("First RK stage produced an invalid state.")
    rhs1 = compute_rhs(stage1, grid, params, domain)
    euler2 = Fields(stage1.rho+dt*rhs1.rho, stage1.vx+dt*rhs1.vx, stage1.vy+dt*rhs1.vy)
    result = _combine(fields, euler2, 0.5, 0.5)
    if np.any(result.rho < 0) or not all(np.isfinite(a).all() for a in (result.rho,result.vx,result.vy)): raise ValueError("RK2 step produced an invalid state.")
    return result

def integrate(fields: Fields, grid: Grid, params: TonerTuParameters, dt: float, n_steps: int, domain: DomainOperators | None = None) -> Fields:
    """Integrate fixed outer steps; domain-specific stability control is separate."""
    if not np.isfinite(dt) or dt <= 0: raise ValueError("Time step dt must be finite and positive.")
    if not isinstance(n_steps,int) or n_steps < 0: raise ValueError("n_steps must be a non-negative integer.")
    current=fields.copy()
    for _ in range(n_steps): current=step(current,grid,params,dt,domain)
    return current