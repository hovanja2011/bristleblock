"""Numerical tests for rectangular geometry and its impermeable walls."""
import numpy as np
import pytest

from mazes import BoxMaze
from toner_tu.initial_conditions import uniform_state
from toner_tu.parameters import TonerTuParameters
from toner_tu.solver import step, integrate


def test_box_domain_conserves_mass_over_many_steps():
    maze = BoxMaze(4.0, 2.0)
    grid = maze.grid(40, 20)
    state = uniform_state(grid, rho0=1.0, vx0=0.2, vy0=-0.1)
    state.rho += 0.02*np.cos(np.pi*grid.x/grid.lx)*np.cos(np.pi*grid.y/grid.ly)
    initial_mass = state.rho.sum()*grid.dx*grid.dy
    params = TonerTuParameters(alpha=0.0, beta=1.0, sigma=0.0, diffusion=0.01)
    result = integrate(state, grid, params, dt=1e-3, n_steps=50, domain=maze.operators())
    final_mass = result.rho.sum()*grid.dx*grid.dy
    assert final_mass == pytest.approx(initial_mass, abs=1e-10)
    assert np.all(result.rho >= 0.0)


def test_box_density_rhs_has_zero_net_flux():
    maze = BoxMaze(3.0, 2.0)
    grid = maze.grid(32, 24)
    state = uniform_state(grid, rho0=1.0, vx0=0.4, vy0=-0.3)
    rhs = maze.operators().density_rhs(state.rho, state.vx, state.vy, grid)
    assert rhs.sum()*grid.dx*grid.dy == pytest.approx(0.0, abs=1e-12)


def test_box_external_normal_face_velocities_are_exactly_zero():
    maze = BoxMaze(2.0, 1.0)
    grid = maze.grid(12, 8)
    rng = np.random.default_rng(17)
    vx = rng.normal(size=(grid.ny, grid.nx))
    vy = rng.normal(size=(grid.ny, grid.nx))
    u_face, v_face = maze.operators().face_velocities(vx, vy)
    assert np.array_equal(u_face[:, 0], np.zeros(grid.ny))
    assert np.array_equal(u_face[:, -1], np.zeros(grid.ny))
    assert np.array_equal(v_face[0, :], np.zeros(grid.nx))
    assert np.array_equal(v_face[-1, :], np.zeros(grid.nx))


def test_box_normal_velocity_uses_odd_reflection_at_wall_faces():
    maze = BoxMaze(2.0, 1.0)
    grid = maze.grid(12, 8)
    ops = maze.operators()
    vx = np.full((grid.ny, grid.nx), 0.7)
    vy = np.full_like(vx, -0.4)
    vx_ghost = ops._ghost(vx, "vx")
    vy_ghost = ops._ghost(vy, "vy")
    assert np.allclose(0.5*(vx[:, 0] + vx_ghost[1:-1, 0]), 0.0)
    assert np.allclose(0.5*(vx[:, -1] + vx_ghost[1:-1, -1]), 0.0)
    assert np.allclose(0.5*(vy[0, :] + vy_ghost[0, 1:-1]), 0.0)
    assert np.allclose(0.5*(vy[-1, :] + vy_ghost[-1, 1:-1]), 0.0)


def test_box_neumann_laplacian_is_second_order_convergent():
    maze = BoxMaze(1.0, 1.0)
    errors = []
    for n in (20, 40, 80):
        grid = maze.grid(n, n)
        x, y = grid.coordinates()
        field = np.cos(np.pi*x)*np.cos(np.pi*y)
        exact = -2.0*np.pi**2*field
        numerical = maze.operators().laplacian(field, grid, "rho")
        errors.append(np.sqrt(np.mean((numerical-exact)**2)))
    assert errors[0]/errors[1] > 3.5
    assert errors[1]/errors[2] > 3.5


def test_box_dimensions_must_be_positive_and_finite():
    for dimensions in [(0, 2), (-1, 2), (np.inf, 2), (2, np.nan)]:
        with pytest.raises(ValueError):
            BoxMaze(*dimensions)
