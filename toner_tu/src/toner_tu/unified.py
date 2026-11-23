from dataclasses import dataclass

import numpy as np

from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.maze_geometry import MazeGeometry


@dataclass(frozen=True)
class UnifiedBoundaryConditions:
    """Outer boundary options. Periodic applies in both coordinate directions."""
    periodic: bool = True
    rho_in: float = 1.0
    vx_in: float = 0.0
    vy_in: float = 0.0

    def __post_init__(self):
        if self.rho_in < 0:
            raise ValueError("rho_in must be non-negative")
        if not np.isfinite([self.rho_in, self.vx_in, self.vy_in]).all():
            raise ValueError("Boundary values must be finite")


def _validate(fields: Fields, grid: Grid, geometry: MazeGeometry):
    geometry.validate_for(grid)
    shape = (grid.ny, grid.nx)
    for name, value in (("rho", fields.rho), ("vx", fields.vx), ("vy", fields.vy)):
        if value.shape != shape:
            raise ValueError(f"{name} shape must be {shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")
    if np.any(fields.rho[geometry.solid] != 0):
        raise ValueError("Density must be zero in solid cells")
    if np.any(fields.rho < 0):
        raise ValueError("Density must be non-negative")


def _neighbor(a, axis, direction, periodic, solid):
    shifted = np.roll(a, direction, axis=axis)
    shifted_solid = np.roll(solid, direction, axis=axis)
    shifted = np.where(shifted_solid, a, shifted)
    if not periodic:
        if axis == 1:
            if direction == 1:
                shifted[:, 0] = a[:, 0]
            else:
                shifted[:, -1] = a[:, -1]
        else:
            if direction == 1:
                shifted[0, :] = a[0, :]
            else:
                shifted[-1, :] = a[-1, :]
    return shifted


def _derivatives(a, grid, geometry, periodic):
    xp = _neighbor(a, 1, -1, periodic, geometry.solid)
    xm = _neighbor(a, 1, 1, periodic, geometry.solid)
    yp = _neighbor(a, 0, -1, periodic, geometry.solid)
    ym = _neighbor(a, 0, 1, periodic, geometry.solid)
    gx = (xp - xm) / (2 * grid.dx)
    gy = (yp - ym) / (2 * grid.dy)
    lap = (xp - 2*a + xm) / grid.dx**2 + (yp - 2*a + ym) / grid.dy**2
    gx[geometry.solid] = 0
    gy[geometry.solid] = 0
    lap[geometry.solid] = 0
    return gx, gy, lap


def _density_flux_rhs(fields, grid, geometry, boundary):
    rho, vx, vy = fields.rho, fields.vx, fields.vy
    fluid = geometry.fluid
    fx = np.zeros((grid.ny, grid.nx + 1))
    fy = np.zeros((grid.ny + 1, grid.nx))

    uf = 0.5 * (vx[:, :-1] + vx[:, 1:])
    open_x = fluid[:, :-1] & fluid[:, 1:]
    fx[:, 1:-1] = np.where(
        open_x,
        np.maximum(uf, 0) * rho[:, :-1] + np.minimum(uf, 0) * rho[:, 1:],
        0.0,
    )
    vf = 0.5 * (vy[:-1, :] + vy[1:, :])
    open_y = fluid[:-1, :] & fluid[1:, :]
    fy[1:-1, :] = np.where(
        open_y,
        np.maximum(vf, 0) * rho[:-1, :] + np.minimum(vf, 0) * rho[1:, :],
        0.0,
    )

    if boundary.mode == "periodic":
        u_left = 0.5 * (vx[:, -1] + vx[:, 0])
        rho_left = np.where(u_left >= 0, rho[:, -1], rho[:, 0])
        fx[:, 0] = np.where(fluid[:, -1] & fluid[:, 0], u_left * rho_left, 0)
        fx[:, -1] = fx[:, 0]
        v_bottom = 0.5 * (vy[-1, :] + vy[0, :])
        rho_bottom = np.where(v_bottom >= 0, rho[-1, :], rho[0, :])
        fy[0, :] = np.where(fluid[-1, :] & fluid[0, :], v_bottom * rho_bottom, 0)
        fy[-1, :] = fy[0, :]
    elif boundary.mode == "box":
        # Closed box: no mass flux through any outer wall.
        pass
    else:
        fx[:, 0] = np.where(fluid[:, 0] & (boundary.vx_in >= 0),
                            boundary.vx_in * boundary.rho_in, 0.0)
        ur = vx[:, -1]
        fx[:, -1] = np.where(fluid[:, -1] & (ur >= 0), ur * rho[:, -1], 0.0)

    drho = -((fx[:, 1:] - fx[:, :-1]) / grid.dx
             + (fy[1:, :] - fy[:-1, :]) / grid.dy)
    drho[geometry.solid] = 0
    return drho


def compute_rhs(fields, grid, params, geometry=None, boundary=None):
    """Unified Toner-Tu RHS for periodic, closed-box, and open-channel boundaries with solid obstacles."""
    if geometry is None:
        geometry = MazeGeometry.empty(grid)
    if boundary is None:
        boundary = UnifiedBoundaryConditions()
    _validate(fields, grid, geometry)

    rho, vx, vy = fields.rho, fields.vx, fields.vy
    drho = _density_flux_rhs(fields, grid, geometry, boundary)
    vx_x, vx_y, lap_vx = _derivatives(vx, grid, geometry, boundary.mode == "periodic")
    vy_x, vy_y, lap_vy = _derivatives(vy, grid, geometry, boundary.periodic)
    rho_x, rho_y, _ = _derivatives(rho, grid, geometry, boundary.periodic)

    adv_vx = vx * vx_x + vy * vx_y
    adv_vy = vx * vy_x + vy * vy_y
    growth = params.alpha - params.beta * (vx**2 + vy**2)
    dvx = (-params.lambda_ * adv_vx + growth * vx
            - params.sigma * rho_x + params.diffusion * lap_vx)
    dvy = (-params.lambda_ * adv_vy + growth * vy
            - params.sigma * rho_y + params.diffusion * lap_vy)

    for value in (drho, dvx, dvy):
        value[geometry.solid] = 0
    return Fields(drho, dvx, dvy)


def step(fields, grid, params, dt, geometry=None, boundary=None):
    """One SSP-RK2 step; rejects invalid intermediate states."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if geometry is None:
        geometry = MazeGeometry.empty(grid)
    if boundary is None:
        boundary = UnifiedBoundaryConditions()

    state0 = fields.copy()
    state0.vx[geometry.solid] = 0.0
    state0.vy[geometry.solid] = 0.0
    _validate(state0, grid, geometry)
    rhs0 = compute_rhs(state0, grid, params, geometry, boundary)
    stage = Fields(state0.rho + dt*rhs0.rho,
                   state0.vx + dt*rhs0.vx,
                   state0.vy + dt*rhs0.vy)
    _validate(stage, grid, geometry)
    rhs1 = compute_rhs(stage, grid, params, geometry, boundary)
    euler2 = Fields(stage.rho + dt*rhs1.rho,
                    stage.vx + dt*rhs1.vx,
                    stage.vy + dt*rhs1.vy)
    _validate(euler2, grid, geometry)
    result = Fields(0.5*state0.rho + 0.5*euler2.rho,
                   0.5*state0.vx + 0.5*euler2.vx,
                   0.5*state0.vy + 0.5*euler2.vy)
    _validate(result, grid, geometry)
    return result


def integrate(fields, grid, params, dt, n_steps, geometry=None, boundary=None):
    if not isinstance(n_steps, int) or n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer")
    current = fields.copy()
    for _ in range(n_steps):
        current = step(current, grid, params, dt, geometry, boundary)
    return current
