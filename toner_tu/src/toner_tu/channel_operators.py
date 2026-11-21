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
) -> Array:
    _check_shape(field, grid)

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
) -> tuple[Array, Array]:
    """Compute the advective derivative using first-order upwind differences."""

    _check_shape(vx, grid)
    _check_shape(vy, grid)

    vx_ext = add_ghost_cells(vx, boundary, "vx")
    vy_ext = add_ghost_cells(vy, boundary, "vy")

    # Neighbouring values in the x direction.
    vx_left = vx_ext[1:-1, :-2]
    vx_right = vx_ext[1:-1, 2:]
    vy_left = vy_ext[1:-1, :-2]
    vy_right = vy_ext[1:-1, 2:]

    # Neighbouring values in the y direction.
    vx_bottom = vx_ext[:-2, 1:-1]
    vx_top = vx_ext[2:, 1:-1]
    vy_bottom = vy_ext[:-2, 1:-1]
    vy_top = vy_ext[2:, 1:-1]

    # Upwind derivatives in x.
    dvx_dx = np.where(
        vx >= 0.0,
        (vx - vx_left) / grid.dx,
        (vx_right - vx) / grid.dx,
    )
    dvy_dx = np.where(
        vx >= 0.0,
        (vy - vy_left) / grid.dx,
        (vy_right - vy) / grid.dx,
    )

    # Upwind derivatives in y.
    dvx_dy = np.where(
        vy >= 0.0,
        (vx - vx_bottom) / grid.dy,
        (vx_top - vx) / grid.dy,
    )
    dvy_dy = np.where(
        vy >= 0.0,
        (vy - vy_bottom) / grid.dy,
        (vy_top - vy) / grid.dy,
    )

    adv_vx = vx * dvx_dx + vy * dvx_dy
    adv_vy = vx * dvy_dx + vy * dvy_dy

    return adv_vx, adv_vy
