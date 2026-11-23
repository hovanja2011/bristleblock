"""Rectangular closed-box geometry and impermeable free-slip operators."""
from dataclasses import dataclass
import numpy as np
from toner_tu.fields import Grid


@dataclass(frozen=True)
class BoxMaze:
    """Empty rectangular container with four impermeable outer walls."""
    width: float
    height: float

    def __post_init__(self) -> None:
        dimensions = np.asarray([self.width, self.height], dtype=float)
        if not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
            raise ValueError("Box dimensions must be finite and positive.")

    def grid(self, nx: int, ny: int) -> Grid:
        """Construct a uniform cell-centred grid covering the physical box."""
        return Grid(nx=nx, ny=ny, lx=self.width, ly=self.height)

    def operators(self) -> "BoxDomainOperators":
        """Return the numerical operators imposing closed-box wall conditions."""
        return BoxDomainOperators()


class BoxDomainOperators:
    """Boundary-aware finite differences for a closed rectangular box.

    Scalars and tangential velocity components use even reflection (zero
    normal derivative). The velocity component normal to a wall uses odd
    reflection, placing zero normal velocity at the wall face. Density
    transport uses conservative face fluxes with zero external flux.
    """

    @staticmethod
    def _check_field(field: np.ndarray, grid: Grid, name: str = "field") -> None:
        """Validate a cell-centred field against its physical grid."""
        if not isinstance(field, np.ndarray) or field.shape != (grid.ny, grid.nx):
            raise ValueError(
                f"{name} must have shape {(grid.ny, grid.nx)} for the supplied grid."
            )
        if not np.all(np.isfinite(field)):
            raise ValueError(f"{name} must contain only finite values.")

    @staticmethod
    def _ghost(field: np.ndarray, component: str) -> np.ndarray:
        """Add one reflected ghost-cell layer around a cell-centred field."""
        if component not in {"rho", "vx", "vy"}:
            raise ValueError("component must be 'rho', 'vx' or 'vy'.")
        if field.ndim != 2 or min(field.shape) < 2:
            raise ValueError("field must be 2D with at least 2 cells per axis.")
        ny, nx = field.shape
        ghost = np.empty((ny + 2, nx + 2), dtype=float)
        ghost[1:-1, 1:-1] = field

        # Left/right walls: vx is normal; vy and rho are even-reflected.
        sign_x = -1.0 if component == "vx" else 1.0
        ghost[1:-1, 0] = sign_x * field[:, 0]
        ghost[1:-1, -1] = sign_x * field[:, -1]

        # Bottom/top walls: vy is normal; vx and rho are even-reflected.
        sign_y = -1.0 if component == "vy" else 1.0
        ghost[0, 1:-1] = sign_y * field[0, :]
        ghost[-1, 1:-1] = sign_y * field[-1, :]

        # The five-point stencils do not use corner ghosts, but initialize them.
        ghost[0, 0] = ghost[0, 1]
        ghost[0, -1] = ghost[0, -2]
        ghost[-1, 0] = ghost[-1, 1]
        ghost[-1, -1] = ghost[-1, -2]
        return ghost

    def gradient(self, field: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
        """Second-order centred gradient with homogeneous Neumann walls."""
        self._check_field(field, grid)
        ghost = self._ghost(field, "rho")
        dfdx = (ghost[1:-1, 2:] - ghost[1:-1, :-2]) / (2.0 * grid.dx)
        dfdy = (ghost[2:, 1:-1] - ghost[:-2, 1:-1]) / (2.0 * grid.dy)
        return dfdx, dfdy

    def laplacian(self, field: np.ndarray, grid: Grid, component: str) -> np.ndarray:
        """Five-point Laplacian with component-aware reflected wall values."""
        self._check_field(field, grid)
        ghost = self._ghost(field, component)
        centre = ghost[1:-1, 1:-1]
        d2dx2 = (ghost[1:-1, 2:] - 2.0*centre + ghost[1:-1, :-2]) / grid.dx**2
        d2dy2 = (ghost[2:, 1:-1] - 2.0*centre + ghost[:-2, 1:-1]) / grid.dy**2
        return d2dx2 + d2dy2

    def advective_derivative(self, vx: np.ndarray, vy: np.ndarray, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
        """First-order upwind approximation of (v dot grad)v."""
        self._check_field(vx, grid, "vx")
        self._check_field(vy, grid, "vy")

        def upwind(field: np.ndarray, component: str) -> tuple[np.ndarray, np.ndarray]:
            ghost = self._ghost(field, component)
            dfdx = np.where(vx >= 0.0, (field - ghost[1:-1, :-2])/grid.dx,
                            (ghost[1:-1, 2:] - field)/grid.dx)
            dfdy = np.where(vy >= 0.0, (field - ghost[:-2, 1:-1])/grid.dy,
                            (ghost[2:, 1:-1] - field)/grid.dy)
            return dfdx, dfdy
        dvx_dx, dvx_dy = upwind(vx, "vx")
        dvy_dx, dvy_dy = upwind(vy, "vy")
        return vx*dvx_dx + vy*dvx_dy, vx*dvy_dx + vy*dvy_dy

    def face_velocities(self, vx: np.ndarray, vy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate cell velocities to faces; external normal faces are zero."""
        if vx.shape != vy.shape or vx.ndim != 2:
            raise ValueError("vx and vy must be 2D arrays with identical shapes.")
        if not np.all(np.isfinite(vx)) or not np.all(np.isfinite(vy)):
            raise ValueError("Face velocities require finite cell velocities.")
        ny, nx = vx.shape
        u_face = np.zeros((ny, nx + 1), dtype=float)
        v_face = np.zeros((ny + 1, nx), dtype=float)
        u_face[:, 1:-1] = 0.5*(vx[:, :-1] + vx[:, 1:])
        v_face[1:-1, :] = 0.5*(vy[:-1, :] + vy[1:, :])
        return u_face, v_face

    def density_rhs(self, rho: np.ndarray, vx: np.ndarray, vy: np.ndarray, grid: Grid) -> np.ndarray:
        """Conservative first-order upwind continuity equation with closed walls."""
        self._check_field(rho, grid, "rho")
        self._check_field(vx, grid, "vx")
        self._check_field(vy, grid, "vy")
        if np.any(rho < 0.0):
            raise ValueError("rho must be non-negative.")
        u_face, v_face = self.face_velocities(vx, vy)
        flux_x = np.zeros_like(u_face)
        flux_y = np.zeros_like(v_face)
        flux_x[:, 1:-1] = (np.maximum(u_face[:, 1:-1], 0.0)*rho[:, :-1]
                           + np.minimum(u_face[:, 1:-1], 0.0)*rho[:, 1:])
        flux_y[1:-1, :] = (np.maximum(v_face[1:-1, :], 0.0)*rho[:-1, :]
                           + np.minimum(v_face[1:-1, :], 0.0)*rho[1:, :])
        return -((flux_x[:, 1:] - flux_x[:, :-1])/grid.dx
                 + (flux_y[1:, :] - flux_y[:-1, :])/grid.dy)
