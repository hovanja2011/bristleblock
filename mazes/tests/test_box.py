"""Tests for rectangular maze geometry and impermeable wall operators."""
import numpy as np
import pytest

from mazes import BoxMaze
from toner_tu.initial_conditions import uniform_state
from toner_tu.parameters import TonerTuParameters
from toner_tu.solver import step


def test_box_domain_conserves_mass_in_one_step():
    maze = BoxMaze(4.0, 2.0)
    grid = maze.grid(40, 20)
    domain = maze.operators()
    state = uniform_state(grid, rho0=1.0, vx0=0.2, vy0=-0.1)
    state.rho[:] += 0.02*np.sin(2*np.pi*grid.x/grid.lx)
    mass0 = state.rho.sum()
    result = step(state, grid, TonerTuParameters(0.0, 1.0, 0.0, 0.01),
                  1e-4, domain)
    assert result.rho.sum() == pytest.approx(mass0, abs=1e-10)
    assert np.all(result.rho >= 0)


def test_box_density_rhs_has_zero_net_flux():
    maze = BoxMaze(3.0, 2.0)
    grid = maze.grid(32, 24)
    state = uniform_state(grid, rho0=1.0, vx0=0.4, vy0=-0.3)
    rhs = maze.operators().density_rhs(state.rho, state.vx, state.vy, grid)
    assert rhs.sum() == pytest.approx(0.0, abs=1e-10)


def test_box_wall_reflection_enforces_zero_normal_face_velocity():
    maze = BoxMaze(2.0, 1.0)
    grid = maze.grid(12, 8)
    ops = maze.operators()
    vx = np.full((grid.ny, grid.nx), 0.7)
    vy = np.full_like(vx, -0.4)
    vx_ghost = ops._ghost(vx, "vx")
    vy_ghost = ops._ghost(vy, "vy")
    # Odd reflection makes the interpolated normal velocity at each wall zero.
    assert np.allclose(vx[:, 0] + vx_ghost[1:-1, 0], 0.0)
    assert np.allclose(vx[:, -1] + vx_ghost[1:-1, -1], 0.0)
    assert np.allclose(vy[0, :] + vy_ghost[0, 1:-1], 0.0)
    assert np.allclose(vy[-1, :] + vy_ghost[-1, 1:-1], 0.0)


def test_box_wall_mass_flux_is_zero_at_all_four_sides():
    maze = BoxMaze(3.0, 2.0)
    grid = maze.grid(16, 12)
    state = uniform_state(grid, rho0=1.0, vx0=0.8, vy0=-0.6)
    rhs = maze.operators().density_rhs(state.rho, state.vx, state.vy, grid)
    # Finite-volume flux divergence can redistribute density, but its global
    # sum must vanish because all four outer face fluxes are prescribed zero.
    assert np.sum(rhs) == pytest.approx(0.0, abs=1e-10)


def test_box_dimensions_must_be_positive_and_finite():
    for dimensions in [(0, 2), (-1, 2), (np.inf, 2), (2, np.nan)]:
        with pytest.raises(ValueError):
            BoxMaze(*dimensions)
