"""Rigid-body state and active propulsion parameters for one bristlebot."""

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
        propulsion_force=0.0,
        active_torque=0.0,

        translational_drag=0.0,
        rotational_drag=0.0,

    ):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

        self.theta = float(theta)
        self.omega = float(omega)

        self.radius = radius
        self.mass = mass
        self.moment_of_inertia = 0.5 * mass * radius**2

        self.propulsion_force = propulsion_force
        self.active_torque = active_torque

        self.translational_drag = translational_drag
        self.rotational_drag = rotational_drag

        # Persistent states of robot-wall contacts
        self.wall_contact_states = {}
        self.robot_contact_states = {}


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
