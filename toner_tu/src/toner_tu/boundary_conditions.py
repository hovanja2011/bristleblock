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
        if self.rho_in <= 0:
            raise ValueError("Inlet density must be positive.")


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