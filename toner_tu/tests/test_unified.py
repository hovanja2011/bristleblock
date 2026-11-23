import numpy as np
import pytest

from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.maze_geometry import MazeGeometry
from toner_tu.maze import BoxMaze
from toner_tu.unified import UnifiedBoundaryConditions, compute_rhs, step


def make_fields(grid, rho=1.0, vx=0.0, vy=0.0):
    shape = (grid.ny, grid.nx)
    return Fields(np.full(shape, rho), np.full(shape, vx), np.full(shape, vy))


def test_periodic_density_rhs_conserves_mass():
    grid = Grid(32, 20, 4.0, 2.0)
    fields = make_fields(grid, rho=1.0, vx=0.2, vy=-0.1)
    fields.rho[:] += 0.1 * np.sin(2*np.pi*grid.x/grid.lx)
    rhs = compute_rhs(fields, grid, TonerTuParameters(0.2, 1.0, 0.1, 0.01))
    assert abs(rhs.rho.sum()) < 1e-10


def test_box_has_zero_density_flux_at_all_outer_walls():
    grid = Grid(32, 20, 4.0, 2.0)
    fields = make_fields(grid, rho=1.0, vx=0.4, vy=-0.25)
    fields.rho[:] = 1.0 + 0.1*np.sin(2*np.pi*grid.x/grid.lx)
    rhs = compute_rhs(fields, grid, TonerTuParameters(0, 1, 0, 0),
                      boundary=UnifiedBoundaryConditions(mode="box"))
    assert abs(rhs.rho.sum()) < 1e-10


def test_solid_cells_are_zero_after_step():
    grid = Grid(20, 12, 4.0, 2.0)
    solid = np.zeros((grid.ny, grid.nx), dtype=bool)
    solid[4:8, 8:12] = True
    geometry = MazeGeometry(solid)
    fields = make_fields(grid, rho=1.0, vx=0.1, vy=0.0)
    fields.rho[solid] = 0.0
    result = step(fields, grid, TonerTuParameters(0.1, 1, 0, 0.01),
                  1e-4, geometry)
    assert np.all(result.rho[solid] == 0)
    assert np.all(result.vx[solid] == 0)
    assert np.all(result.vy[solid] == 0)


def test_box_maze_constructs_matching_grid_geometry_and_walls():
    maze = BoxMaze(width=4.0, height=2.0)
    grid = maze.grid(nx=40, ny=20)
    geometry = maze.geometry(grid)
    boundary = maze.boundary_conditions()
    assert geometry.solid.shape == (20, 40)
    assert not geometry.solid.any()
    assert boundary.mode == "box"


def test_box_dimensions_and_grid_must_match():
    with pytest.raises(ValueError):
        BoxMaze(width=0, height=2)
    maze = BoxMaze(width=4, height=2)
    with pytest.raises(ValueError, match="match"):
        maze.geometry(Grid(20, 10, 3, 2))
