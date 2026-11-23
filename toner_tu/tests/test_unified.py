import numpy as np
import pytest

from toner_tu.fields import Fields
from toner_tu.maze import BoxMaze
from toner_tu.parameters import TonerTuParameters
from toner_tu.unified import compute_rhs, step


def make_uniform_fields(grid, rho=1.0, vx=0.0, vy=0.0):
    shape = (grid.ny, grid.nx)
    return Fields(np.full(shape, rho), np.full(shape, vx), np.full(shape, vy))


def test_box_geometry_and_closed_boundary_mass_conservation():
    maze = BoxMaze(width=4.0, height=2.0)
    grid = maze.grid(nx=40, ny=20)
    geometry = maze.geometry(grid)
    boundary = maze.boundary_conditions()

    fields = make_uniform_fields(grid, rho=1.0, vx=0.4, vy=-0.25)
    fields.rho[:] += 0.05 * np.sin(2*np.pi*grid.x/grid.lx)

    rhs = compute_rhs(fields, grid, TonerTuParameters(0.0, 1.0, 0.0, 0.0),
                      geometry, boundary)
    assert geometry.solid.shape == (grid.ny, grid.nx)
    assert not geometry.solid.any()
    assert boundary.mode == "box"
    assert abs(rhs.rho.sum()) < 1e-10


def test_closed_box_step_keeps_density_nonnegative_and_mass():
    maze = BoxMaze(width=4.0, height=2.0)
    grid = maze.grid(nx=24, ny=12)
    geometry = maze.geometry(grid)
    boundary = maze.boundary_conditions()
    fields = make_uniform_fields(grid, rho=1.0, vx=0.0, vy=0.0)
    fields.rho[:] += 0.05*np.cos(2*np.pi*grid.x/grid.lx)
    mass_before = fields.rho.sum()
    result = step(fields, grid, TonerTuParameters(0.0, 1.0, 0.0, 0.01),
                  1e-3, geometry, boundary)
    assert np.all(result.rho >= 0)
    assert result.rho.sum() == pytest.approx(mass_before, abs=1e-10)


def test_box_requires_positive_dimensions_and_matching_grid():
    with pytest.raises(ValueError):
        BoxMaze(width=0, height=2.0)
    maze = BoxMaze(width=4.0, height=2.0)
    with pytest.raises(ValueError, match="match"):
        maze.geometry(maze.grid(nx=20, ny=10).__class__(20, 10, 3.0, 2.0))
