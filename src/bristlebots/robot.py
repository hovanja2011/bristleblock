
import numpy as np


class Robot:
    def __init__(
        self,
        position,
        velocity,

        theta=0.0,
        omega = 0.0,

        radius=1.0,
        mass=1.0,
        propulsion_force=1.0,
        translational_drag=1.0,
        rotational_drag=1.0,
    ):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

        self.theta = float(theta)
        self.omega = float(omega)

        self.radius = radius
        self.mass = mass
        self.moment_of_inertia = 0.5 * mass * radius**2

        self.propulsion_force = propulsion_force
        self.translational_drag = translational_drag
        self.rotational_drag = rotational_drag

    @property
    def direction(self):
        return np.array([
            np.cos(self.theta),
            np.sin(self.theta)
        ])

    @property
    def free_speed(self):
        return (
            self.propulsion_force
            / self.translational_drag
        )
