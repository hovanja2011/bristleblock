import numpy as np

from bristlebots.contacts import (
    ContactState,
    ContactParameters,
    robot_wall_contact,
)


def compute_forces(robot):
    propulsion = (
        robot.propulsion_force
        * robot.direction
    )

    drag = (
        -robot.translational_drag
        * robot.velocity
    )

    return propulsion + drag


def compute_torque(robot):
    return (
        -robot.rotational_drag
        * robot.omega
    )


def step(robot, dt, walls=None, contact_parameters=None):
    """
    Advance robot dynamics by one time step.

    Includes:
        - active propulsion
        - translational drag
        - rotational drag
        - robot-wall contact forces
        - robot-wall contact torques
    """

    if dt <= 0:
        raise ValueError("Time step must be positive.")

    if walls is None:
        walls = []

    if walls and contact_parameters is None:
        raise ValueError(
            "Contact parameters must be provided when walls are present."
        )

    # Free-motion forces
    force = compute_forces(robot)
    torque = compute_torque(robot)

    # Sum contact forces and torques from all walls
    for wall in walls:

        if wall not in robot.wall_contact_states:
            robot.wall_contact_states[wall] = ContactState()

        state = robot.wall_contact_states[wall]

        contact_force, contact_torque = robot_wall_contact(
            robot,
            wall,
            dt,
            state,
            contact_parameters,
        )

        force += contact_force
        torque += contact_torque

    # Translational dynamics
    acceleration = force / robot.mass

    robot.velocity += acceleration * dt
    robot.position += robot.velocity * dt

    # Rotational dynamics
    angular_acceleration = torque / robot.moment_of_inertia

    robot.omega += angular_acceleration * dt
    robot.theta += robot.omega * dt