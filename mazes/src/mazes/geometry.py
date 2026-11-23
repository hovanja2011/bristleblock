"""Analytic signed-distance geometry for rounded maze obstacles."""
from dataclasses import dataclass
import numpy as np
from toner_tu.fields import Grid


def _point(value, name):
    point = np.asarray(value, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must be a finite 2D point.")
    return tuple(float(v) for v in point)


@dataclass(frozen=True)
class CircleObstacle:
    """Circular solid obstacle described by centre and radius."""
    center: tuple[float, float]
    radius: float
    def __post_init__(self):
        object.__setattr__(self, "center", _point(self.center, "center"))
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("Obstacle radius must be finite and positive.")
    def signed_distance(self, x, y):
        return np.hypot(x-self.center[0], y-self.center[1])-self.radius


@dataclass(frozen=True)
class RoundedSegment:
    """Finite-thickness segment with circular end caps (a capsule)."""
    start: tuple[float, float]
    end: tuple[float, float]
    radius: float
    def __post_init__(self):
        object.__setattr__(self, "start", _point(self.start, "start"))
        object.__setattr__(self, "end", _point(self.end, "end"))
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("Wall half-thickness must be finite and positive.")
        if np.linalg.norm(np.asarray(self.end)-np.asarray(self.start)) <= 0.0:
            raise ValueError("RoundedSegment endpoints must be distinct.")
    def signed_distance(self, x, y):
        return _segment_distance(x, y, self.start, self.end)-self.radius


def _segment_distance(x, y, start, end):
    ax, ay = start
    bx, by = end
    dx, dy = bx-ax, by-ay
    projection = np.clip(((x-ax)*dx+(y-ay)*dy)/(dx*dx+dy*dy), 0.0, 1.0)
    return np.hypot(x-(ax+projection*dx), y-(ay+projection*dy))


@dataclass(frozen=True)
class RoundedPolyline:
    """Rounded wall centreline: union of capsules along a polyline.

    A bend is rounded by the disk-radius Minkowski thickening of its vertex;
    this avoids a sharp zero-radius corner in the solid-wall geometry.
    """
    points: tuple
    radius: float
    def __post_init__(self):
        points = tuple(_point(p, "polyline point") for p in self.points)
        if len(points) < 2:
            raise ValueError("RoundedPolyline requires at least two points.")
        if any(np.linalg.norm(np.asarray(b)-np.asarray(a)) <= 0.0
               for a, b in zip(points[:-1], points[1:])):
            raise ValueError("Consecutive polyline points must be distinct.")
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("Wall half-thickness must be finite and positive.")
        object.__setattr__(self, "points", points)
    def signed_distance(self, x, y):
        distances = [_segment_distance(x, y, a, b) for a, b in zip(self.points[:-1], self.points[1:])]
        return np.minimum.reduce(distances)-self.radius


@dataclass(frozen=True)
class MazeGeometry:
    """Rectangular container with analytic circular and rounded wall solids."""
    width: float
    height: float
    obstacles: tuple = ()
    def __post_init__(self):
        dimensions = np.asarray([self.width, self.height], dtype=float)
        if not np.isfinite(dimensions).all() or np.any(dimensions <= 0.0):
            raise ValueError("Maze dimensions must be finite and positive.")
        object.__setattr__(self, "obstacles", tuple(self.obstacles))
        valid = (CircleObstacle, RoundedSegment, RoundedPolyline)
        if not all(isinstance(o, valid) for o in self.obstacles):
            raise TypeError("Unsupported obstacle geometry.")
    def grid(self, nx: int, ny: int) -> Grid:
        return Grid(nx=nx, ny=ny, lx=self.width, ly=self.height)
    def signed_distance(self, x, y):
        """Signed distance to the union of obstacles (negative inside solids)."""
        if not self.obstacles:
            return np.full_like(np.asarray(x, dtype=float), np.inf)
        return np.minimum.reduce([o.signed_distance(x, y) for o in self.obstacles])
    def solid_mask(self, grid: Grid):
        x, y = grid.coordinates()
        return self.signed_distance(x, y) <= 0.0
    def operators(self, grid: Grid):
        from mazes.operators import MazeDomainOperators
        return MazeDomainOperators(self, grid)
