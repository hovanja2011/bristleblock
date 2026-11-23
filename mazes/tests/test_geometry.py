"""Tests for analytic rounded obstacles and conservative no-flux operators."""
import numpy as np
import pytest
from mazes import CircleObstacle, MazeGeometry, RoundedSegment
from toner_tu.initial_conditions import uniform_state
from toner_tu.parameters import TonerTuParameters
from toner_tu.solver import integrate

def test_circle_signed_distance_is_analytic():
    circle = CircleObstacle((2.0, 3.0), 0.5)
    assert circle.signed_distance(2.0, 3.0) == pytest.approx(-0.5)
    assert circle.signed_distance(2.5, 3.0) == pytest.approx(0.0)
    assert circle.signed_distance(3.0, 3.0) == pytest.approx(0.5)

def test_rounded_segment_has_circular_end_caps():
    wall = RoundedSegment((1.0, 2.0), (4.0, 2.0), 0.25)
    assert wall.signed_distance(2.0, 2.0) == pytest.approx(-0.25)
    assert wall.signed_distance(1.0, 2.0) == pytest.approx(-0.25)
    assert wall.signed_distance(1.0, 2.25) == pytest.approx(0.0)
    assert wall.signed_distance(0.75, 2.0) == pytest.approx(0.0)

def test_maze_mask_and_flux_are_impermeable_and_mass_conservative():
    geometry = MazeGeometry(4.0, 3.0, obstacles=(
        CircleObstacle((2.0, 1.5), 0.45),
        RoundedSegment((0.8, 0.8), (3.0, 0.8), 0.12),
    ))
    grid = geometry.grid(80, 60)
    operators = geometry.operators(grid)
    mask = operators.solid_mask
    assert mask.any() and (~mask).any()
    state = operators.mask_fields(uniform_state(grid, rho0=1.0, vx0=0.2, vy0=0.1))
    assert np.all(state.rho[mask] == 0.0)
    assert np.all(state.vx[mask] == 0.0)
    assert np.all(state.vy[mask] == 0.0)
    rhs = operators.density_rhs(state.rho, state.vx, state.vy, grid)
    assert np.sum(rhs)*grid.dx*grid.dy == pytest.approx(0.0, abs=1e-12)
    assert np.all(rhs[mask] == 0.0)
    params = TonerTuParameters(alpha=0.0, beta=1.0, sigma=0.0, diffusion=0.0, lambda_=0.0)
    mass0 = np.sum(state.rho)*grid.dx*grid.dy
    result = integrate(state, grid, params, dt=1e-4, n_steps=10, domain=operators)
    mass1 = np.sum(result.rho)*grid.dx*grid.dy
    assert mass1 == pytest.approx(mass0, abs=1e-10)
    assert np.all(result.rho >= 0.0)
    assert np.all(result.rho[mask] == 0.0)

def test_rounded_obstacle_mask_area_converges_with_grid_refinement():
    geometry = MazeGeometry(2.0, 2.0, (CircleObstacle((1.0, 1.0), 0.4),))
    exact_area = np.pi*0.4**2
    errors = []
    for n in (40, 80, 160):
        grid = geometry.grid(n, n)
        area = np.count_nonzero(geometry.solid_mask(grid))*grid.dx*grid.dy
        errors.append(abs(area-exact_area))
    assert errors[-1] < errors[0]
