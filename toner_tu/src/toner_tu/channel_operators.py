import numpy as np
import numpy.typing as npt

from toner_tu.fields import Grid
from toner_tu.boundary_conditions import (
    ChannelBoundaryConditions,
    add_ghost_cells,
)


Array = npt.NDArray[np.float64]


def _check_shape(field: Array, grid: Grid) -> None:
    expected = (grid.ny, grid.nx)

    if field.shape != expected:
        raise ValueError(
            f"Expected shape {expected}, got {field.shape}."
        )


def gradient(
    field: Array,
    grid: Grid,
    boundary: ChannelBoundaryConditions,
    field_name: str,
) -> tuple[Array, Array]:
    _check_shape(field, grid)

    extended = add_ghost_cells(field, boundary, field_name)

    dfdx = (
        extended[1:-1, 2:] - extended[1:-1, :-2]
    ) / (2.0 * grid.dx)

    dfdy = (
        extended[2:, 1:-1] - extended[:-2, 1:-1]
    ) / (2.0 * grid.dy)

    return dfdx, dfdy

def laplacian(
    field: Array,
    grid: Grid,
    boundary: ChannelBoundaryConditions,
    field_name: str,
    extended: Array | None = None,
) -> Array:
    _check_shape(field, grid)

    if extended is None:
        extended = add_ghost_cells(field, boundary, field_name)

    center = extended[1:-1, 1:-1]

    d2fdx2 = (
        extended[1:-1, 2:]
        - 2.0 * center
        + extended[1:-1, :-2]
    ) / grid.dx**2

    d2fdy2 = (
        extended[2:, 1:-1]
        - 2.0 * center
        + extended[:-2, 1:-1]
    ) / grid.dy**2

    return d2fdx2 + d2fdy2


def advective_derivative(
    vx: Array,
    vy: Array,
    grid: Grid,
    boundary: ChannelBoundaryConditions,
) -> tuple[Array, Array]:
    _check_shape(vx, grid)
    _check_shape(vy, grid)

    dvx_dx, dvx_dy = gradient(vx, grid, boundary, "vx")
    dvy_dx, dvy_dy = gradient(vy, grid, boundary, "vy")

    adv_vx = vx * dvx_dx + vy * dvx_dy
    adv_vy = vx * dvy_dx + vy * dvy_dy

    return adv_vx, adv_vy



def advective_derivative_upwind(
    vx: Array,
    vy: Array,
    grid: Grid,
    boundary: ChannelBoundaryConditions,
    vx_extended: Array | None = None,
    vy_extended: Array | None = None,
) -> tuple[Array, Array]:
    """Compute the advective derivative using first-order upwind differences."""

    _check_shape(vx, grid)
    _check_shape(vy, grid)

    if vx_extended is None:
        vx_extended = add_ghost_cells(vx, boundary, "vx")

    if vy_extended is None:
        vy_extended = add_ghost_cells(vy, boundary, "vy")

    vx_ext = vx_extended
    vy_ext = vy_extended

    vx_left = vx_ext[1:-1, :-2]
    vx_right = vx_ext[1:-1, 2:]
    vy_left = vy_ext[1:-1, :-2]
    vy_right = vy_ext[1:-1, 2:]

    vx_bottom = vx_ext[:-2, 1:-1]
    vx_top = vx_ext[2:, 1:-1]
    vy_bottom = vy_ext[:-2, 1:-1]
    vy_top = vy_ext[2:, 1:-1]

    dvx_dx = np.empty_like(vx)
    dvy_dx = np.empty_like(vy)
    dvx_dy = np.empty_like(vx)
    dvy_dy = np.empty_like(vy)
    temporary = np.empty_like(vx)

    # X derivatives.
    np.subtract(vx, vx_left, out=dvx_dx)
    np.divide(dvx_dx, grid.dx, out=dvx_dx)

    np.subtract(vx_right, vx, out=temporary)
    np.divide(temporary, grid.dx, out=temporary)
    np.copyto(dvx_dx, temporary, where=(vx < 0.0))

    np.subtract(vy, vy_left, out=dvy_dx)
    np.divide(dvy_dx, grid.dx, out=dvy_dx)

    np.subtract(vy_right, vy, out=temporary)
    np.divide(temporary, grid.dx, out=temporary)
    np.copyto(dvy_dx, temporary, where=(vx < 0.0))

    # Y derivatives.
    np.subtract(vx, vx_bottom, out=dvx_dy)
    np.divide(dvx_dy, grid.dy, out=dvx_dy)

    np.subtract(vx_top, vx, out=temporary)
    np.divide(temporary, grid.dy, out=temporary)
    np.copyto(dvx_dy, temporary, where=(vy < 0.0))

    np.subtract(vy, vy_bottom, out=dvy_dy)
    np.divide(dvy_dy, grid.dy, out=dvy_dy)

    np.subtract(vy_top, vy, out=temporary)
    np.divide(temporary, grid.dy, out=temporary)
    np.copyto(dvy_dy, temporary, where=(vy < 0.0))

    adv_vx = vx * dvx_dx + vy * dvx_dy
    adv_vy = vx * dvy_dx + vy * dvy_dy

    return adv_vx, adv_vy
