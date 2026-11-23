"""Reference spatial finite differences for a doubly-periodic Cartesian grid.

These operators define the bulk/reference problem. Non-periodic boundaries and
obstacles are implemented in the separate mazes project through DomainOperators.
"""
import numpy as np
import numpy.typing as npt
from toner_tu.fields import Grid

Array = npt.NDArray[np.float64]


def _check_shape(field: Array, grid: Grid) -> None:
    """Reject arrays that do not match the grid's [ny, nx] layout."""
    if field.shape != (grid.ny, grid.nx):
        raise ValueError(f"Expected shape {(grid.ny, grid.nx)}, got {field.shape}.")


def gradient(field: Array, grid: Grid) -> tuple[Array, Array]:
    """Second-order centred scalar gradient with periodic wraparound."""
    _check_shape(field, grid)
    dfdx = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2.0 * grid.dx)
    dfdy = (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2.0 * grid.dy)
    return dfdx, dfdy


def divergence(vx: Array, vy: Array, grid: Grid) -> Array:
    """Divergence of a vector field, assembled from periodic derivatives."""
    _check_shape(vx, grid)
    _check_shape(vy, grid)
    dvx_dx, _ = gradient(vx, grid)
    _, dvy_dy = gradient(vy, grid)
    return dvx_dx + dvy_dy


def laplacian(field: Array, grid: Grid) -> Array:
    """Five-point second-order Laplacian with periodic neighbours."""
    _check_shape(field, grid)
    d2fdx2 = (np.roll(field, -1, axis=1)-2.0*field+np.roll(field, 1, axis=1))/grid.dx**2
    d2fdy2 = (np.roll(field, -1, axis=0)-2.0*field+np.roll(field, 1, axis=0))/grid.dy**2
    return d2fdx2 + d2fdy2


def advective_derivative(vx: Array, vy: Array, grid: Grid) -> tuple[Array, Array]:
    """Return the two components of (v dot grad)v using centred derivatives."""
    _check_shape(vx, grid)
    _check_shape(vy, grid)
    dvx_dx, dvx_dy = gradient(vx, grid)
    dvy_dx, dvy_dy = gradient(vy, grid)
    return vx*dvx_dx + vy*dvx_dy, vx*dvy_dx + vy*dvy_dy


def vorticity(vx: Array, vy: Array, grid: Grid) -> Array:
    """Scalar 2D vorticity omega_z = d(vy)/dx - d(vx)/dy."""
    _check_shape(vx, grid)
    _check_shape(vy, grid)
    _, dvx_dy = gradient(vx, grid)
    dvy_dx, _ = gradient(vy, grid)
    return dvy_dx - dvx_dy
