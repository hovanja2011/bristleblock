from dataclasses import dataclass

import numpy as np

from toner_tu.fields import Fields


@dataclass(frozen=True)
class ChannelBoundaryConditions:
    """Boundary values for an open straight channel."""

    rho_in: float = 1.0
    vx_in: float = 0.5
    vy_in: float = 0.0

    def __post_init__(self):
        if self.rho_in < 0:
            raise ValueError("rho_in must be non-negative")


def apply_channel_boundary_conditions(
    fields: Fields,
    boundary: ChannelBoundaryConditions,
) -> Fields:
    """
    Apply approximate channel boundary conditions.

    Left: prescribed inlet values.
    Right: zero-gradient outlet.
    Top and bottom: impermeable free-slip walls.

    The arrays are assumed to be ordered as [y, x].
    """

    if fields.rho.shape[0] < 3 or fields.rho.shape[1] < 3:
        raise ValueError("Channel grid must have at least 3 cells per axis.")

    result = fields.copy()

    # Inlet: prescribed density and velocity.
    result.rho[:, 0] = boundary.rho_in
    result.vx[:, 0] = boundary.vx_in
    result.vy[:, 0] = boundary.vy_in

    # Outlet: zero normal gradient.
    result.rho[:, -1] = result.rho[:, -2]
    result.vx[:, -1] = result.vx[:, -2]
    result.vy[:, -1] = result.vy[:, -2]

    # Bottom wall: no penetration and free slip.
    result.rho[0, :] = result.rho[1, :]
    result.vx[0, :] = result.vx[1, :]
    result.vy[0, :] = 0.0

    # Top wall: no penetration and free slip.
    result.rho[-1, :] = result.rho[-2, :]
    result.vx[-1, :] = result.vx[-2, :]
    result.vy[-1, :] = 0.0

    return result


def add_ghost_cells(
    field: np.ndarray,
    boundary: ChannelBoundaryConditions,
    field_name: str,
) -> np.ndarray:
    """
    Extend a cell-centered field with one ghost cell on each side.
    """

    if field.ndim != 2:
        raise ValueError("Field must be two-dimensional.")

    ny, nx = field.shape

    if ny < 2 or nx < 2:
        raise ValueError("Field must have at least 2 cells per axis.")

    result = np.empty((ny + 2, nx + 2), dtype=np.float64)

    # Copy physical cells.
    result[1:-1, 1:-1] = field

    # Left boundary: inlet.
    if field_name == "rho":
        inlet_value = boundary.rho_in
    elif field_name == "vx":
        inlet_value = boundary.vx_in
    elif field_name == "vy":
        inlet_value = boundary.vy_in
    else:
        raise ValueError(f"Unknown field: {field_name}")

    result[1:-1, 0] = 2.0 * inlet_value - field[:, 0]

    # Right boundary: zero normal gradient.
    result[1:-1, -1] = field[:, -1]

    # Bottom boundary: free-slip / no penetration.
    result[0, 1:-1] = field[0, :]
    if field_name == "vy":
        result[0, 1:-1] = -field[0, :]

    # Top boundary: free-slip / no penetration.
    result[-1, 1:-1] = field[-1, :]
    if field_name == "vy":
        result[-1, 1:-1] = -field[-1, :]

    # Corners: wall condition has priority.
    result[0, 0] = result[0, 1]
    result[0, -1] = result[0, -2]
    result[-1, 0] = result[-1, 1]
    result[-1, -1] = result[-1, -2]

    return result