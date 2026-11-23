import numpy as np
import pytest

from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.maze_geometry import MazeGeometry
from toner_tu.unified import UnifiedBoundaryConditions, compute_rhs, step


def make_fields(grid, rho=1.0, vx=0.0, vy=0.0):
    shape = (grid.ny, grid.nx)
    return Fields(np.full(shape, rho), np.full(shape, vx), np.full(shape, vy))


def test_unified_periodic_density_mass_conservation():
    grid = Grid(32, 20, 4.0, 2.0)
    fields = make_fields(grid, rho=1.0, vx=0.2, vy=-0.1)
    fields.rho[:] += 0.1 * np.sin(2*np.pi*grid.x/grid.lx)
    rhs = compute_rhs(fields, grid, TonerTuParameters(0.2, 1.0, 0.1, 0.01))
    assert abs(rhs.rho.sum()) < 1e-10


def test_mass_flux_is_blocked_by_solid_column():
    grid = Grid(24, 16, 4.0, 2.0)
    solid = np.zeros((grid.ny, grid.nx), dtype=bool)
    solid[:, 12] = True
    geometry = MazeGeometry(solid)
    fields = make_fields(grid, rho=1.0, vx=0.5)
    fields.rho[:, :12] = 2.0
    fields.rho[:, 12:] = 0.5
    fields.rho[solid] = 0
    rhs = compute_rhs(fields, grid, TonerTuParameters(0, 1, 0, 0),
                      geometry, UnifiedBoundaryConditions(periodic=True))
    assert abs(rhs.rho[:, :12].sum() + rhs.rho[:, 13:].sum()) < 1e-10


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


def test_geometry_shape_is_checked():
    grid = Grid(12, 8, 3, 2)
    geometry = MazeGeometry(np.zeros((8, 11), dtype=bool))
    with pytest.raises(ValueError, match="Geometry shape"):
        geometry.validate_for(grid)


def test_solid_mask_must_be_boolean():
    with pytest.raises(TypeError, match="boolean"):
        MazeGeometry(np.zeros((4, 5), dtype=int))
