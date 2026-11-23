"""Geometric primitives used to represent fixed physical walls."""

import numpy as np


class Wall:
    """A finite, immobile line segment representing a wall."""

    def __init__(self, start, end):
        self.start = np.array(start, dtype=float)
        self.end = np.array(end, dtype=float)

        self.length = np.linalg.norm(self.end - self.start)

        if self.length == 0:
            raise ValueError("Wall must have non-zero length.")

        self.direction = (self.end - self.start) / self.length

    def closest_point(self, point):
        """Return the closest point on the wall segment to a point."""
        point = np.asarray(point, dtype=float)

        relative = point - self.start
        projection = np.dot(relative, self.direction)

        # Clamp the projection to the finite segment.
        projection = np.clip(projection, 0.0, self.length)

        return self.start + projection * self.direction

    def contact_geometry(self, position, radius):
        """
        Compute geometry of contact between a disk and the wall.

        Returns:
            distance: distance from disk center to wall
            normal: unit vector pointing from wall to disk center
            overlap: positive overlap, zero if not in contact
        """
        closest = self.closest_point(position)
        displacement = np.asarray(position, dtype=float) - closest

        distance = np.linalg.norm(displacement)
        overlap = radius - distance

        if distance > 1e-12:
            normal = displacement / distance
        else:
            # Fallback for a center exactly on the wall.
            normal = np.array([-self.direction[1], self.direction[0]])

        return distance, normal, max(overlap, 0.0)
