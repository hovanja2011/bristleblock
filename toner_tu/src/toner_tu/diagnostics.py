"""Global diagnostics for continuum active-matter fields."""
import numpy as np
from toner_tu.fields import Fields, Grid


def total_mass(fields: Fields, grid: Grid) -> float:
    """Integral of density over the computational domain."""
    return float(np.sum(fields.rho) * grid.dx * grid.dy)


def mean_density(fields: Fields) -> float:
    """Spatial mean density."""
    return float(np.mean(fields.rho))


def minimum_density(fields: Fields) -> float:
    """Minimum density, useful for detecting loss of positivity."""
    return float(np.min(fields.rho))


def mean_velocity(fields: Fields) -> tuple[float, float]:
    """Spatially averaged velocity components."""
    return float(np.mean(fields.vx)), float(np.mean(fields.vy))


def mean_speed(fields: Fields) -> float:
    """Mean magnitude of the local velocity."""
    return float(np.mean(np.sqrt(fields.vx**2 + fields.vy**2)))


def polarization(fields: Fields) -> float:
    """Global polar order P = |<v>|/<|v|>; returns zero for a motionless state."""
    speed = np.sqrt(fields.vx**2 + fields.vy**2)
    denominator = float(np.mean(speed))
    if denominator == 0.0:
        return 0.0
    vx_mean, vy_mean = float(np.mean(fields.vx)), float(np.mean(fields.vy))
    return float(np.sqrt(vx_mean**2 + vy_mean**2) / denominator)


def kinetic_energy_density(fields: Fields) -> float:
    """Mean kinetic-energy density under the unit-mass-density convention."""
    return float(0.5*np.mean(fields.rho*(fields.vx**2 + fields.vy**2)))
