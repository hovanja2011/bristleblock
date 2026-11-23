from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from toner_tu.fields import Grid


@dataclass(frozen=True)
class MazeGeometry:
    """Cell-based geometry: True cells are solid, False cells are fluid."""
    solid: npt.NDArray[np.bool_]

    def __post_init__(self) -> None:
        if self.solid.ndim != 2:
            raise ValueError("solid mask must be two-dimensional")
        if self.solid.dtype != np.bool_:
            raise TypeError("solid mask must have boolean dtype")

    @classmethod
    def empty(cls, grid: Grid) -> "MazeGeometry":
        return cls(np.zeros((grid.ny, grid.nx), dtype=bool))

    @classmethod
    def from_rectangles(cls, grid: Grid, rectangles) -> "MazeGeometry":
        """Rasterize axis-aligned rectangles (x0, y0, x1, y1) at cell centers."""
        x, y = grid.coordinates()
        solid = np.zeros((grid.ny, grid.nx), dtype=bool)
        for x0, y0, x1, y1 in rectangles:
            if x1 <= x0 or y1 <= y0:
                raise ValueError("Rectangle extents must be positive")
            solid |= (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        return cls(solid)

    def validate_for(self, grid: Grid) -> None:
        expected = (grid.ny, grid.nx)
        if self.solid.shape != expected:
            raise ValueError(f"Geometry shape must be {expected}, got {self.solid.shape}")
        if np.all(self.solid):
            raise ValueError("Geometry contains no fluid cells")

    @property
    def fluid(self) -> npt.NDArray[np.bool_]:
        return ~self.solid
