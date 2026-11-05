from dataclasses import dataclass

import numpy as np


@dataclass
class ContactParameters:
    normal_stiffness: float
    normal_damping: float
    mu_static: float
    mu_kinetic: float
    tangential_stiffness: float
    tangential_damping: float = 0.0

    def __post_init__(self):
        if self.normal_stiffness <= 0:
            raise ValueError("Normal stiffness must be positive.")

        if self.tangential_stiffness <= 0:
            raise ValueError("Tangential stiffness must be positive.")

        if self.normal_damping < 0 or self.tangential_damping < 0:
            raise ValueError("Damping coefficients cannot be negative.")

        if self.mu_kinetic < 0 or self.mu_static < 0:
            raise ValueError("Friction coefficients cannot be negative.")

        if self.mu_static < self.mu_kinetic:
            raise ValueError(
                "Static friction coefficient must be >= kinetic friction coefficient."
            )


@dataclass
class ContactState:
    tangential_displacement: float = 0.0


def robot_wall_contact(robot, wall, dt, state, parameters):
    """
    Calculate normal and tangential contact forces between
    a circular robot and an immobile wall.

    Tangential friction follows the Cundall-Strack model.

    Returns:
        force: total contact force on the robot
        torque: contact torque around the robot center
    """

    if dt <= 0:
        raise ValueError("Time step must be positive.")

    _, normal, overlap = wall.contact_geometry(
        robot.position,
        robot.radius,
    )

    # No contact: erase the contact history.
    if overlap <= 0.0:
        state.tangential_displacement = 0.0
        return np.zeros(2), 0.0

    # Normal contact force
    normal_velocity = np.dot(robot.velocity, normal)

    fn = (
        parameters.normal_stiffness * overlap
        - parameters.normal_damping * normal_velocity
    )

    fn = max(fn, 0.0)

    # If there is no compressive normal force,
    # there can be no friction force.
    if fn <= 0.0:
        state.tangential_displacement = 0.0
        return np.zeros(2), 0.0

    normal_force = fn * normal

    # Contact point relative to robot center
    r_contact = -robot.radius * normal

    # Velocity of the contact point
    rotational_velocity = robot.omega * np.array([
        -r_contact[1],
        r_contact[0],
    ])

    contact_velocity = robot.velocity + rotational_velocity

    # Tangential direction
    tangent = np.array([-normal[1], normal[0]])

    tangential_velocity = np.dot(contact_velocity, tangent)

    # Update tangential displacement history
    state.tangential_displacement += tangential_velocity * dt

    # Trial friction force
    trial_force = (
        -parameters.tangential_stiffness
        * state.tangential_displacement
        -parameters.tangential_damping * tangential_velocity
    )

    # Coulomb friction limit
    static_limit = parameters.mu_static * fn

    if abs(trial_force) <= static_limit:
        # Sticking contact
        tangential_force_magnitude = trial_force

    else:
        # Sliding contact
        if abs(tangential_velocity) > 1e-14:
            slip_direction = np.sign(tangential_velocity)
        else:
            slip_direction = np.sign(trial_force)

        tangential_force_magnitude = (
            -parameters.mu_kinetic * fn * slip_direction
        )

        # Correct the stored displacement so that the
        # spring force does not exceed the kinetic limit.
        state.tangential_displacement = (
            -tangential_force_magnitude
            - parameters.tangential_damping * tangential_velocity
        ) / parameters.tangential_stiffness

    tangential_force = tangential_force_magnitude * tangent

    # Total force
    force = normal_force + tangential_force

    # Torque around robot center
    torque = (
        r_contact[0] * tangential_force[1]
        - r_contact[1] * tangential_force[0]
    )

    return force, torque