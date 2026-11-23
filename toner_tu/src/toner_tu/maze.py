"""High-level maze layouts and their outer boundary-condition choices."""

from dataclasses import dataclass

from toner_tu.fields import Grid
from toner_tu.maze_geometry import MazeGeometry
from toner_tu.unified import UnifiedBoundaryConditions


@dataclass(frozen=True)
class BoxMaze:
    """An empty rectangular maze enclosed by four impermeable outer walls."""

    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Box dimensions must be positive")

    def grid(self, nx: int, ny: int) -> Grid:
        return Grid(nx=nx, ny=ny, lx=self.width, ly=self.height)

    def geometry(self, grid: Grid) -> MazeGeometry:
        if grid.lx != self.width or grid.ly != self.height:
            raise ValueError("Grid domain must match the box dimensions")
        return MazeGeometry.empty(grid)

    def boundary_conditions(self) -> UnifiedBoundaryConditions:
        return UnifiedBoundaryConditions(mode="box")
