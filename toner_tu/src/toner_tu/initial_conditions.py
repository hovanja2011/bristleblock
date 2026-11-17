
import numpy as np

from toner_tu.fields import Fields, Grid


def uniform_state(
    grid: Grid,
    rho0: float = 1.0,
    vx0: float = 0.0,
    vy0: float = 0.0,
) -> Fields:
    """Create a spatially uniform initial state."""

    if rho0 <= 0:
        raise ValueError("Initial density must be positive.")

    shape = (grid.ny, grid.nx)

    return Fields(
        rho=np.full(shape, rho0, dtype=float),
        vx=np.full(shape, vx0, dtype=float),
        vy=np.full(shape, vy0, dtype=float),
    )


def random_velocity_state(
    grid: Grid,
    rho0: float = 1.0,
    velocity_scale: float = 0.1,
    seed: int | None = None,
) -> Fields:
    """Create uniform density with random initial velocity."""

    if rho0 <= 0:
        raise ValueError("Initial density must be positive.")

    if velocity_scale < 0:
        raise ValueError("Velocity scale must be non-negative.")

    rng = np.random.default_rng(seed)
    shape = (grid.ny, grid.nx)

    return Fields(
        rho=np.full(shape, rho0, dtype=float),
        vx=rng.normal(0.0, velocity_scale, size=shape),
        vy=rng.normal(0.0, velocity_scale, size=shape),
    )
