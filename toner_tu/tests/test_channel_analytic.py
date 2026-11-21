
import numpy as np

from toner_tu.channel_operators import laplacian
from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.boundary_conditions import ChannelBoundaryConditions
from toner_tu.channel_solver import integrate
from toner_tu.boundary_conditions import add_ghost_cells

def test_uniform_velocity_relaxation_matches_analytic_solution():
    grid = Grid(nx=16, ny=8, lx=20.0, ly=10.0)

    alpha = 1.0
    beta = 1.0
    v0 = 0.2
    final_time = 0.1

    params = TonerTuParameters(
        alpha=alpha,
        beta=beta,
        sigma=0.0,
        diffusion=0.0,
        lambda_=0.0,
    )

    boundary = ChannelBoundaryConditions(
        rho_in=1.0,
        vx_in=v0,
        vy_in=0.0,
    )

    initial = Fields(
        rho=np.ones((grid.ny, grid.nx)),
        vx=np.full((grid.ny, grid.nx), v0),
        vy=np.zeros((grid.ny, grid.nx)),
    )

    numerical = integrate(
        initial,
        grid,
        params,
        boundary,
        dt=0.001,
        n_steps=100,
    )

    # Exact solution of dv/dt = (alpha - beta*v^2)*v
    coefficient = alpha / (beta * v0**2) - 1.0

    exact_velocity = np.sqrt(
        (alpha / beta)
        / (1.0 + coefficient * np.exp(-2.0 * alpha * final_time))
    )

    np.testing.assert_allclose(
        numerical.vx,
        exact_velocity,
        rtol=2e-5,
        atol=1e-8,
    )

    np.testing.assert_allclose(
        numerical.vy,
        0.0,
        atol=1e-10,
    )

    assert np.all(np.isfinite(numerical.rho))
    assert np.all(numerical.rho >= 0.0)


def analytical_speed(t, v0, alpha, beta):
    """Exact solution of dv/dt = (alpha - beta*v^2)*v."""
    denominator = (
        1.0
        + (alpha / (beta * v0**2) - 1.0)
        * np.exp(-2.0 * alpha * t)
    )
    return np.sqrt((alpha / beta) / denominator)


def test_homogeneous_reaction_converges_to_analytical_solution():
    grid = Grid(nx=8, ny=8, lx=8.0, ly=8.0)

    alpha = 1.0
    beta = 1.0
    v0 = 0.5
    final_time = 0.2

    params = TonerTuParameters(
        alpha=alpha,
        beta=beta,
        sigma=0.0,
        diffusion=0.0,
        lambda_=0.0,
    )

    boundary = ChannelBoundaryConditions(
        rho_in=0.0,
        vx_in=v0,
        vy_in=0.0,
    )

    shape = (grid.ny, grid.nx)

    initial = Fields(
        rho=np.zeros(shape),
        vx=np.full(shape, v0),
        vy=np.zeros(shape),
    )

    errors = []

    for dt in (0.01, 0.005):
        n_steps = round(final_time / dt)

        result = integrate(
            initial,
            grid,
            params,
            boundary,
            dt,
            n_steps,
        )

        exact = analytical_speed(
            final_time, v0, alpha, beta
        )

        numerical = np.mean(result.vx)
        error = abs(numerical - exact)
        errors.append(error)

        # Homogeneous density and velocity direction must be preserved.
        assert np.allclose(result.rho, 0.0, atol=1e-11)
        assert np.allclose(result.vy, 0.0, atol=1e-11)
        assert np.allclose(result.vx, numerical, atol=1e-11)

    coarse_error, fine_error = errors

    assert fine_error < coarse_error

    observed_order = np.log2(coarse_error / fine_error)

    assert 1.8 < observed_order < 2.2, (
        f"Expected second-order temporal convergence, "
        f"got order {observed_order:.4f}; "
        f"errors: {errors}"
    )

def test_laplacian_matches_discrete_neumann_eigenmode():
    grid = Grid(nx=16, ny=16, lx=8.0, ly=8.0)
    boundary = ChannelBoundaryConditions(
        rho_in=1.0,
        vx_in=1.0,
        vy_in=0.0,
    )

    mode = 1
    k = mode * np.pi / grid.ly

    y = grid.y[:, 0]
    field = np.broadcast_to(
        np.cos(k * y)[:, None],
        (grid.ny, grid.nx),
    ).copy()

    discrete_eigenvalue = (
        -4.0 * np.sin(k * grid.dy / 2.0) ** 2
        / grid.dy**2
    )

    numerical = laplacian(field, grid, boundary, "vx")
    analytical = discrete_eigenvalue * field

    # Exclude x-boundary columns: the test isolates the
    # y-direction Neumann eigenmode.
    assert np.allclose(
        numerical[:, 1:-1],
        analytical[:, 1:-1],
        rtol=1e-11,
        atol=1e-11,
    )


def test_diffusion_damps_velocity_perturbation():
    grid = Grid(nx=32, ny=32, lx=8.0, ly=8.0)

    diffusion = 0.1
    amplitude = 1e-3
    final_time = 0.1

    params = TonerTuParameters(
        alpha=0.0,
        beta=1.0,
        sigma=0.0,
        diffusion=diffusion,
        lambda_=0.0,
    )

    boundary = ChannelBoundaryConditions(
        rho_in=1.0,
        vx_in=0.0,
        vy_in=0.0,
    )

    x = grid.x
    k = np.pi / (2.0 * grid.lx)
    mode = np.sin(k * x)

    shape = (grid.ny, grid.nx)

    initial = Fields(
        rho=np.ones(shape),
        vx=amplitude * mode,
        vy=np.zeros(shape),
    )

    dt = 0.001
    n_steps = round(final_time / dt)

    result = integrate(
        initial,
        grid,
        params,
        boundary,
        dt,
        n_steps,
    )

    discrete_eigenvalue = (
        -4.0 * np.sin(k * grid.dx / 2.0) ** 2
        / grid.dx**2
    )

    expected_amplitude = amplitude * np.exp(
        diffusion * discrete_eigenvalue * final_time
    )

    measured_amplitude = (
        np.sum(result.vx * mode)
        / np.sum(mode**2)
    )

    assert abs(measured_amplitude) < amplitude

    assert np.isclose(
        measured_amplitude,
        expected_amplitude,
        rtol=2e-3,
        atol=1e-10,
    )


def test_channel_velocity_boundary_conditions():
    grid = Grid(nx=64, ny=16, lx=16.0, ly=4.0)

    params = TonerTuParameters(
        alpha=0.0,
        beta=1.0,
        sigma=0.0,
        diffusion=0.05,
        lambda_=1.0,
    )

    boundary = ChannelBoundaryConditions(
        rho_in=1.0,
        vx_in=0.5,
        vy_in=0.0,
    )

    shape = (grid.ny, grid.nx)

    x = grid.x
    y = grid.y

    # Smooth localized perturbation moving towards the outlet.
    perturbation = 0.05 * np.exp(
        -((x - 8.0) ** 2) / 2.0
    )

    initial = Fields(
        rho=np.ones(shape),
        vx=0.5 + perturbation,
        vy=np.zeros(shape),
    )

    dt = 0.001
    n_steps = 1000

    result = integrate(
        initial,
        grid,
        params,
        boundary,
        dt,
        n_steps,
    )

    # 1. Inlet: prescribed velocity is maintained.
    vx_extended = add_ghost_cells(
        result.vx,
        boundary,
        "vx",
    )

    # Inlet boundary face.
    inlet_face_value = 0.5 * (
        vx_extended[1:-1, 0] + vx_extended[1:-1, 1]
    )

    assert np.allclose(
        inlet_face_value,
        boundary.vx_in,
        atol=1e-12,
    )

    # Outlet zero-gradient condition.
    outlet_gradient = (
        vx_extended[1:-1, -1]
        - vx_extended[1:-1, -2]
    ) / grid.dx

    assert np.allclose(
        outlet_gradient,
        0.0,
        atol=1e-12,
    )

    # 2. Outlet: zero normal gradient.
    outlet_gradient = (
        result.vx[:, -1] - result.vx[:, -2]
    ) / grid.dx

    # Inspect the ghost cells used by the actual channel operators.
    vx_extended = add_ghost_cells(
        result.vx,
        boundary,
        "vx",
    )

    # 1. Inlet: the value at the boundary face is prescribed.
    inlet_face_value = 0.5 * (
        vx_extended[1:-1, 0] + vx_extended[1:-1, 1]
    )

    assert np.allclose(
        inlet_face_value,
        boundary.vx_in,
        atol=1e-12,
    )

    # 2. Outlet: zero normal gradient in the ghost-cell extension.
    outlet_gradient = (
        vx_extended[1:-1, -1]
        - vx_extended[1:-1, -2]
    ) / grid.dx

    assert np.allclose(
        outlet_gradient,
        0.0,
        atol=1e-12,
    )

    # 3. The solution remains finite and physically admissible.
    assert np.all(np.isfinite(result.rho))
    assert np.all(np.isfinite(result.vx))
    assert np.all(np.isfinite(result.vy))
    assert np.all(result.rho >= 0.0)
