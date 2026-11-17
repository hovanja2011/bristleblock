
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from toner_tu.fields import Grid


@dataclass(frozen=True)
class Channel:
    """Straight rectangular channel."""

    grid: Grid

    @property
    def x(self) -> npt.NDArray:
        return np.arange(self.grid.nx) * self.grid.dx

    @property
    def y(self) -> npt.NDArray:
        return np.arange(self.grid.ny) * self.grid.dy

    def coordinates(self):
        return np.meshgrid(self.x, self.y)

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.ny, self.grid.nx

    def is_inside(self, x, y) -> bool:
        return (
            0.0 <= x <= self.grid.lx
            and 0.0 <= y <= self.grid.ly
        )
