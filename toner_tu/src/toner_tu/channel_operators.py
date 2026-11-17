
import numpy as np
import numpy.typing as npt

from toner_tu.fields import Grid


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
) -> tuple[Array, Array]:
    """
    Gradient with non-periodic boundary conditions.

    Central differences in the interior.
    First-order one-sided differences at boundaries.
    """

    _check_shape(field, grid)

    dfdx = np.empty_like(field)
    dfdy = np.empty_like(field)

    # x derivative
    dfdx[:, 1:-1] = (
        field[:, 2:] - field[:, :-2]
    ) / (2.0 * grid.dx)

    dfdx[:, 0] = (
        field[:, 1] - field[:, 0]
    ) / grid.dx

    dfdx[:, -1] = (
        field[:, -1] - field[:, -2]
    ) / grid.dx

    # y derivative
    dfdy[1:-1, :] = (
        field[2:, :] - field[:-2, :]
    ) / (2.0 * grid.dy)

    dfdy[0, :] = (
        field[1, :] - field[0, :]
    ) / grid.dy

    dfdy[-1, :] = (
        field[-1, :] - field[-2, :]
    ) / grid.dy

    return dfdx, dfdy


def divergence(
    vx: Array,
    vy: Array,
    grid: Grid,
) -> Array:
    """Divergence of a vector field."""

    _check_shape(vx, grid)
    _check_shape(vy, grid)

    dvx_dx, _ = gradient(vx, grid)
    _, dvy_dy = gradient(vy, grid)

    return dvx_dx + dvy_dy


def laplacian(
    field: Array,
    grid: Grid,
) -> Array:
    """
    Laplacian with non-periodic finite differences.

    Second-order central differences in the interior.
    Second-order one-sided approximation at boundaries.
    """

    _check_shape(field, grid)

    result = np.empty_like(field)

    # x contribution
    d2x = np.empty_like(field)

    d2x[:, 1:-1] = (
        field[:, 2:]
        - 2.0 * field[:, 1:-1]
        + field[:, :-2]
    ) / grid.dx**2

    d2x[:, 0] = (
        2.0 * field[:, 0]
        - 5.0 * field[:, 1]
        + 4.0 * field[:, 2]
        - field[:, 3]
    ) / grid.dx**2

    d2x[:, -1] = (
        2.0 * field[:, -1]
        - 5.0 * field[:, -2]
        + 4.0 * field[:, -3]
        - field[:, -4]
    ) / grid.dx**2

    # y contribution
    d2y = np.empty_like(field)

    d2y[1:-1, :] = (
        field[2:, :]
        - 2.0 * field[1:-1, :]
        + field[:-2, :]
    ) / grid.dy**2

    d2y[0, :] = (
        2.0 * field[0, :]
        - 5.0 * field[1, :]
        + 4.0 * field[2, :]
        - field[3, :]
    ) / grid.dy**2

    d2y[-1, :] = (
        2.0 * field[-1, :]
        - 5.0 * field[-2, :]
        + 4.0 * field[-3, :]
        - field[-4, :]
    ) / grid.dy**2

    result[:] = d2x + d2y

    return result


def advective_derivative(
    vx: Array,
    vy: Array,
    grid: Grid,
) -> tuple[Array, Array]:
    """Calculate (v · grad) v."""

    _check_shape(vx, grid)
    _check_shape(vy, grid)

    dvx_dx, dvx_dy = gradient(vx, grid)
    dvy_dx, dvy_dy = gradient(vy, grid)

    adv_vx = vx * dvx_dx + vy * dvx_dy
    adv_vy = vx * dvy_dx + vy * dvy_dy

    return adv_vx, adv_vy
