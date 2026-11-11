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

    # Rotational drag
    torque = -robot.rotational_drag * robot.omega

    # Rotate propulsion offset from body frame to world frame
    c = np.cos(robot.theta)
    s = np.sin(robot.theta)

    x_body, y_body = robot.propulsion_offset

    r_x = c * x_body - s * y_body
    r_y = s * x_body + c * y_body

    # Propulsion force in world coordinates
    force_x = robot.propulsion_force * c
    force_y = robot.propulsion_force * s

    # 2D cross product: r x F
    propulsion_torque = (
        r_x * force_y - r_y * force_x
    )

    torque += propulsion_torque

    return torque


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