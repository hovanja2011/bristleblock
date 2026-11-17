
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    lx: float
    ly: float

    def __post_init__(self) -> None:
        if self.nx < 2 or self.ny < 2:
            raise ValueError("Grid dimensions must be at least 2.")

        if self.lx <= 0 or self.ly <= 0:
            raise ValueError("Domain lengths must be positive.")

    @property
    def dx(self) -> float:
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        return self.ly / self.ny

    def coordinates(self) -> tuple[npt.NDArray, npt.NDArray]:
        x = np.arange(self.nx) * self.dx
        y = np.arange(self.ny) * self.dy
        return np.meshgrid(x, y)


@dataclass
class Fields:
    rho: npt.NDArray[np.float64]
    vx: npt.NDArray[np.float64]
    vy: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.rho.ndim != 2:
            raise ValueError("Fields must be two-dimensional.")

        if self.rho.shape != self.vx.shape or self.rho.shape != self.vy.shape:
            raise ValueError("All fields must have the same shape.")

    def copy(self) -> "Fields":
        return Fields(
            rho=self.rho.copy(),
            vx=self.vx.copy(),
            vy=self.vy.copy(),
        )
