
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


def compute_upwind_fluxes(rho, vx, vy, grid, boundary):
    """
    Compute conservative upwind mass fluxes at cell faces.

    Left boundary:
        Prescribed inlet velocity and density for inflow.
        Interior density is used for outflow.

    Right boundary:
        Interior density is used for outflow.
        Inflow from outside is zero (vacuum).

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

    # Left boundary.
    # Use the prescribed boundary velocity, not vx[:, 0].
    u_left = boundary.vx_in

    flux_x[:, 0] = np.where(
        u_left >= 0.0,
        u_left * boundary.rho_in,
        u_left * rho[:, 0],
    )

    # Right boundary.
    # Outflow is allowed; external inflow has zero density.
    u_right = u[:, -1]

    flux_x[:, -1] = np.where(
        u_right >= 0.0,
        u_right * rho[:, -1],
        0.0,
    )

    # Internal horizontal faces.
    v_internal = v[1:-1, :]

    flux_y[1:-1, :] = (
        np.maximum(v_internal, 0.0) * rho[:-1, :]
        + np.minimum(v_internal, 0.0) * rho[1:, :]
    )

    # Top and bottom fluxes remain zero.

    return flux_x, flux_y



def compute_density_rhs(rho, vx, vy, grid, boundary):
    """
    Conservative finite-volume approximation of

        d rho / dt = - div(rho * velocity).

    Returns the density time derivative.
    """

    flux_x, flux_y = compute_upwind_fluxes(
        rho, vx, vy, grid, boundary
    )

    divergence = (
        (flux_x[:, 1:] - flux_x[:, :-1]) / grid.dx
        + (flux_y[1:, :] - flux_y[:-1, :]) / grid.dy
    )

    return -divergence

def density_cfl_rate(vx, vy, grid, boundary):
    """
    Return the maximum outgoing mass-flux rate per cell.

    This rate gives the forward-Euler positivity condition:
        dt * rate <= 1.

    The prescribed left boundary velocity is used.
    External inflow at the right boundary has zero density.
    """

    u = compute_x_face_velocities(vx)
    v = compute_y_face_velocities(vy)

    u[:, 0] = boundary.vx_in

    outgoing_x = (
        np.maximum(u[:, 1:], 0.0)
        + np.maximum(-u[:, :-1], 0.0)
    ) / grid.dx

    outgoing_y = (
        np.maximum(v[1:, :], 0.0)
        + np.maximum(-v[:-1, :], 0.0)
    ) / grid.dy

    outgoing_rate = outgoing_x + outgoing_y

    return float(np.max(outgoing_rate))