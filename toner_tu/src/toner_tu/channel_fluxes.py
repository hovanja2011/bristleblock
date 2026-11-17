
import numpy as np

from toner_tu.fields import Grid


def _check_fields(rho, vx, vy, grid):
    expected_shape = (grid.ny, grid.nx)

    if rho.shape != expected_shape:
        raise ValueError("rho has an incorrect shape")

    if vx.shape != expected_shape or vy.shape != expected_shape:
        raise ValueError("velocity fields have incorrect shapes")

    if not (
        np.all(np.isfinite(rho))
        and np.all(np.isfinite(vx))
        and np.all(np.isfinite(vy))
    ):
        raise ValueError("Fields must contain only finite values")

    if np.any(rho < 0):
        raise ValueError("Density must be non-negative")


def compute_x_face_velocities(vx):
    """
    Cell-centered vx -> velocities at vertical cell faces.

    Boundary velocities are extrapolated from the nearest cell.
    """

    ny, nx = vx.shape
    u = np.zeros((ny, nx + 1))

    u[:, 1:-1] = 0.5 * (vx[:, :-1] + vx[:, 1:])
    u[:, 0] = vx[:, 0]
    u[:, -1] = vx[:, -1]

    return u


def compute_y_face_velocities(vy):
    """
    Cell-centered vy -> velocities at horizontal cell faces.

    The top and bottom boundaries are impermeable.
    """

    ny, nx = vy.shape
    v = np.zeros((ny + 1, nx))

    v[1:-1, :] = 0.5 * (vy[:-1, :] + vy[1:, :])

    # v[0, :] = 0 and v[-1, :] = 0:
    # no normal flow through the channel walls.

    return v


def compute_upwind_fluxes(rho, vx, vy, grid):
    """
    Compute conservative upwind mass fluxes at cell faces.

    Left and right boundaries are open with zero exterior density:
    particles may leave, but no particles enter from outside.

    Top and bottom boundaries are impermeable.
    """

    _check_fields(rho, vx, vy, grid)

    u = compute_x_face_velocities(vx)
    v = compute_y_face_velocities(vy)

    ny, nx = rho.shape

    flux_x = np.zeros((ny, nx + 1))
    flux_y = np.zeros((ny + 1, nx))

    # Internal vertical faces.
    u_internal = u[:, 1:-1]

    flux_x[:, 1:-1] = (
        np.maximum(u_internal, 0.0) * rho[:, :-1]
        + np.minimum(u_internal, 0.0) * rho[:, 1:]
    )

    # Left boundary:
    # positive velocity would bring particles from outside.
    # Exterior density is zero, so only outward flux is possible.
    flux_x[:, 0] = np.minimum(u[:, 0], 0.0) * rho[:, 0]

    # Right boundary:
    # only positive velocity transports particles out of the domain.
    flux_x[:, -1] = np.maximum(u[:, -1], 0.0) * rho[:, -1]

    # Internal horizontal faces.
    v_internal = v[1:-1, :]

    flux_y[1:-1, :] = (
        np.maximum(v_internal, 0.0) * rho[:-1, :]
        + np.minimum(v_internal, 0.0) * rho[1:, :]
    )

    # Top and bottom fluxes remain zero.

    return flux_x, flux_y


def compute_density_rhs(rho, vx, vy, grid):
    """
    Conservative finite-volume approximation of

        d rho / dt = - div(rho * velocity).

    Returns the density time derivative.
    """

    flux_x, flux_y = compute_upwind_fluxes(
        rho, vx, vy, grid
    )

    divergence = (
        (flux_x[:, 1:] - flux_x[:, :-1]) / grid.dx
        + (flux_y[1:, :] - flux_y[:-1, :]) / grid.dy
    )

    return -divergence


def density_cfl_rate(vx, vy, grid):
    """
    Return the maximum outgoing mass-flux rate per cell.

    A sufficient positivity condition for forward Euler is:

        dt * density_cfl_rate <= 1.
    """

    u = compute_x_face_velocities(vx)
    v = compute_y_face_velocities(vy)

    outgoing_x = (
        np.maximum(u[:, 1:], 0.0)
        + np.maximum(-u[:, :-1], 0.0)
    ) / grid.dx

    outgoing_y = (
        np.maximum(v[1:, :], 0.0)
        + np.maximum(-v[:-1, :], 0.0)
    ) / grid.dy

    return float(np.max(outgoing_x + outgoing_y))
