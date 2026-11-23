"""Conservative finite-volume transport and masked operators for maze domains."""
import numpy as np
from toner_tu.fields import Fields, Grid

class MazeDomainOperators:
    """Grid-resolved no-through-flow operators for internal rounded obstacles.

    Shapes are analytic signed-distance functions sampled at cell centres.
    A face shared with a solid cell is impermeable. This conservative
    Cartesian mask method is not a cut-cell or immersed-boundary scheme.
    Velocity diffusion uses a zero-normal-gradient, free-slip-like mask closure.
    """
    def __init__(self, geometry, grid: Grid):
        if grid.lx != geometry.width or grid.ly != geometry.height:
            raise ValueError("Grid and maze geometry dimensions must match.")
        self.geometry, self.grid = geometry, grid
        self._solid = geometry.solid_mask(grid)
    @property
    def solid_mask(self):
        return self._solid.copy()
    def mask_fields(self, fields: Fields) -> Fields:
        """Copy a state and set density and velocity to zero inside solids."""
        self._check(fields.rho, "rho")
        result = fields.copy()
        for array in (result.rho, result.vx, result.vy):
            array[self._solid] = 0.0
        return result
    def _check(self, field, name):
        if not isinstance(field, np.ndarray) or field.shape != (self.grid.ny, self.grid.nx):
            raise ValueError(f"{name} must match maze grid {(self.grid.ny, self.grid.nx)}.")
        if not np.isfinite(field).all():
            raise ValueError(f"{name} must contain finite values.")
    def _neighbor(self, field, dy, dx):
        neighbour = np.roll(field, shift=(-dy, -dx), axis=(0, 1))
        solid_neighbour = np.roll(self._solid, shift=(-dy, -dx), axis=(0, 1))
        if dx == 1:
            neighbour[:, -1], solid_neighbour[:, -1] = field[:, -1], True
        elif dx == -1:
            neighbour[:, 0], solid_neighbour[:, 0] = field[:, 0], True
        if dy == 1:
            neighbour[-1, :], solid_neighbour[-1, :] = field[-1, :], True
        elif dy == -1:
            neighbour[0, :], solid_neighbour[0, :] = field[0, :], True
        return np.where(solid_neighbour, field, neighbour)
    def gradient(self, field, grid):
        self._check(field, "field")
        if grid != self.grid:
            raise ValueError("Operators were constructed for a different grid.")
        gx = (self._neighbor(field, 0, 1)-self._neighbor(field, 0, -1))/(2*grid.dx)
        gy = (self._neighbor(field, 1, 0)-self._neighbor(field, -1, 0))/(2*grid.dy)
        gx[self._solid] = 0.0
        gy[self._solid] = 0.0
        return gx, gy
    def laplacian(self, field, grid, component="rho"):
        self._check(field, component)
        if grid != self.grid:
            raise ValueError("Operators were constructed for a different grid.")
        right, left = self._neighbor(field, 0, 1), self._neighbor(field, 0, -1)
        up, down = self._neighbor(field, 1, 0), self._neighbor(field, -1, 0)
        result = ((right-2*field+left)/grid.dx**2
                  + (up-2*field+down)/grid.dy**2)
        result[self._solid] = 0.0
        return result
    def advective_derivative(self, vx, vy, grid):
        self._check(vx, "vx")
        self._check(vy, "vy")
        if grid != self.grid:
            raise ValueError("Operators were constructed for a different grid.")
        def upwind(field):
            left, right = self._neighbor(field, 0, -1), self._neighbor(field, 0, 1)
            down, up = self._neighbor(field, -1, 0), self._neighbor(field, 1, 0)
            dx = np.where(vx >= 0, (field-left)/grid.dx, (right-field)/grid.dx)
            dy = np.where(vy >= 0, (field-down)/grid.dy, (up-field)/grid.dy)
            return dx, dy
        vx_x, vx_y = upwind(vx)
        vy_x, vy_y = upwind(vy)
        ax, ay = vx*vx_x+vy*vx_y, vx*vy_x+vy*vy_y
        ax[self._solid] = 0.0
        ay[self._solid] = 0.0
        return ax, ay
    def density_rhs(self, rho, vx, vy, grid):
        self._check(rho, "rho")
        self._check(vx, "vx")
        self._check(vy, "vy")
        if grid != self.grid:
            raise ValueError("Operators were constructed for a different grid.")
        if np.any(rho < 0.0):
            raise ValueError("Density must be non-negative.")
        ny, nx = rho.shape
        u_face, v_face = np.zeros((ny, nx+1)), np.zeros((ny+1, nx))
        fluid_x = ~(self._solid[:, :-1] | self._solid[:, 1:])
        fluid_y = ~(self._solid[:-1, :] | self._solid[1:, :])
        u_face[:, 1:-1] = np.where(fluid_x, 0.5*(vx[:, :-1]+vx[:, 1:]), 0.0)
        v_face[1:-1, :] = np.where(fluid_y, 0.5*(vy[:-1, :]+vy[1:, :]), 0.0)
        flux_x, flux_y = np.zeros_like(u_face), np.zeros_like(v_face)
        flux_x[:, 1:-1] = (np.maximum(u_face[:, 1:-1], 0)*rho[:, :-1]
                           + np.minimum(u_face[:, 1:-1], 0)*rho[:, 1:])
        flux_y[1:-1, :] = (np.maximum(v_face[1:-1, :], 0)*rho[:-1, :]
                           + np.minimum(v_face[1:-1, :], 0)*rho[1:, :])
        rhs = -((flux_x[:, 1:]-flux_x[:, :-1])/grid.dx
                + (flux_y[1:, :]-flux_y[:-1, :])/grid.dy)
        rhs[self._solid] = 0.0
        return rhs
