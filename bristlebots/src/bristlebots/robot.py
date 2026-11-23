"""Rigid-body state and active propulsion parameters for one bristlebot."""

import numpy as np


def _vector2(value, name):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 2D vector.")
    return vector.copy()


class Robot:
    """Circular rigid robot with central active propulsion and linear drag.

    The propulsion force acts through the centre of mass and therefore
    contributes no direct torque. active_torque is an independently specified
    applied torque.
    """

    def __init__(
        self,
        position,
        velocity,
        theta=0.0,
        omega=0.0,
        radius=1.0,
        mass=1.0,
        propulsion_force=0.0,
        active_torque=0.0,
        translational_drag=0.0,
        rotational_drag=0.0,
    ):
        self.position = _vector2(position, "position")
        self.velocity = _vector2(velocity, "velocity")

        scalar_values = np.asarray(
            [theta, omega, radius, mass, propulsion_force, active_torque,
             translational_drag, rotational_drag],
            dtype=float,
        )
        if not np.all(np.isfinite(scalar_values)):
            raise ValueError("Robot parameters must be finite.")
        if radius <= 0.0 or mass <= 0.0:
            raise ValueError("Robot radius and mass must be positive.")
        if translational_drag < 0.0 or rotational_drag < 0.0:
            raise ValueError("Drag coefficients must be non-negative.")

        self.theta = float(theta)
        self.omega = float(omega)
        self.radius = float(radius)
        self.mass = float(mass)
        self.moment_of_inertia = 0.5 * self.mass * self.radius**2

        self.propulsion_force = float(propulsion_force)
        self.active_torque = float(active_torque)
        self.translational_drag = float(translational_drag)
        self.rotational_drag = float(rotational_drag)

        self.wall_contact_states = {}
        self.robot_contact_states = {}

    @property
    def direction(self):
        """Unit vector along the body heading."""
        return np.array([np.cos(self.theta), np.sin(self.theta)])

    @property
    def propulsion_force_vector(self):
        """Active propulsion force in world coordinates."""
        return self.propulsion_force * self.direction

    @property
    def free_speed(self):
        """Terminal speed under linear drag; infinity if driven without drag."""
        if self.translational_drag == 0.0:
            return float("inf") if self.propulsion_force != 0.0 else 0.0
        return self.propulsion_force / self.translational_drag
