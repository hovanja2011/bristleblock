"""Geometric primitives used to represent fixed physical walls."""

import numpy as np


class Wall:
    """A finite, immobile line segment representing a wall."""

    def __init__(self, start, end):
        self.start = np.asarray(start, dtype=float)
        self.end = np.asarray(end, dtype=float)
        if (self.start.shape != (2,) or self.end.shape != (2,)
                or not np.all(np.isfinite(self.start))
                or not np.all(np.isfinite(self.end))):
            raise ValueError("Wall endpoints must be finite 2D vectors.")

        displacement = self.end - self.start
        self.length = float(np.linalg.norm(displacement))
        if not np.isfinite(self.length) or self.length <= 0.0:
            raise ValueError("Wall must have non-zero finite length.")

        self.direction = displacement / self.length

    def closest_point(self, point):
        """Return the closest point on the finite wall segment to a point."""
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("point must be a finite 2D vector.")

        relative = point - self.start
        projection = np.dot(relative, self.direction)
        projection = np.clip(projection, 0.0, self.length)
        return self.start + projection * self.direction

    def contact_geometry(self, position, radius):
        """Return centre-to-wall distance, wall-to-centre normal and overlap."""
        position = np.asarray(position, dtype=float)
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            raise ValueError("position must be a finite 2D vector.")
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be finite and positive.")

        closest = self.closest_point(position)
        displacement = position - closest
        distance = float(np.linalg.norm(displacement))
        overlap = radius - distance

        if distance > 1e-12:
            normal = displacement / distance
        else:
            # The side is geometrically ambiguous when the centre lies exactly
            # on the segment; choose its left-hand normal deterministically.
            normal = np.array([-self.direction[1], self.direction[0]])

        return distance, normal, max(overlap, 0.0)
