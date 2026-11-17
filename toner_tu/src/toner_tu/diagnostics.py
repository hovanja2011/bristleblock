
import numpy as np

from toner_tu.fields import Fields, Grid


def total_mass(fields: Fields, grid: Grid) -> float:
    return float(np.sum(fields.rho) * grid.dx * grid.dy)


def mean_density(fields: Fields) -> float:
    return float(np.mean(fields.rho))


def minimum_density(fields: Fields) -> float:
    return float(np.min(fields.rho))


def mean_velocity(fields: Fields) -> tuple[float, float]:
    return (
        float(np.mean(fields.vx)),
        float(np.mean(fields.vy)),
    )


def mean_speed(fields: Fields) -> float:
    speed = np.sqrt(fields.vx**2 + fields.vy**2)
    return float(np.mean(speed))


def polarization(fields: Fields) -> float:
    """
    Polarization based on the spatially averaged velocity.

    P = |<v>| / <|v|>
    """
    speed = np.sqrt(fields.vx**2 + fields.vy**2)
    denominator = float(np.mean(speed))

    if denominator == 0.0:
        return 0.0

    vx_mean = float(np.mean(fields.vx))
    vy_mean = float(np.mean(fields.vy))

    return float(np.sqrt(vx_mean**2 + vy_mean**2) / denominator)


def kinetic_energy_density(fields: Fields) -> float:
    """Mean kinetic energy density for unit mass density convention."""
    return float(
        0.5 * np.mean(fields.rho * (fields.vx**2 + fields.vy**2))
    )
