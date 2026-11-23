"""Interface between Toner–Tu physics and domain-specific spatial operators."""
from typing import Protocol
from toner_tu.fields import Grid

class DomainOperators(Protocol):
    """Boundary-aware spatial operators required by the continuum model."""
    def density_rhs(self, rho, vx, vy, grid: Grid): ...
    def gradient(self, field, grid: Grid): ...
    def laplacian(self, field, grid: Grid, component: str): ...
    def advective_derivative(self, vx, vy, grid: Grid): ...

class PeriodicDomain:
    """Reference periodic domain for bulk validation and compatibility."""
    def density_rhs(self, rho, vx, vy, grid):
        from toner_tu.operators import divergence
        return -divergence(rho * vx, rho * vy, grid)
    def gradient(self, field, grid):
        from toner_tu.operators import gradient
        return gradient(field, grid)
    def laplacian(self, field, grid, component="rho"):
        from toner_tu.operators import laplacian
        return laplacian(field, grid)
    def advective_derivative(self, vx, vy, grid):
        from toner_tu.operators import advective_derivative
        return advective_derivative(vx, vy, grid)