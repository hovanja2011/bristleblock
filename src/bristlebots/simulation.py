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
    Return force and torque diagnostics calculated before integration.
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
    propulsion_force = (
        robot.propulsion_force * robot.direction
    )

    drag_force = (
        -robot.translational_drag * robot.velocity
    )

    # Free-motion torques
    rotational_drag_torque = (
        -robot.rotational_drag * robot.omega
    )

    c = np.cos(robot.theta)
    s = np.sin(robot.theta)

    x_body, y_body = robot.propulsion_offset

    r_x = c * x_body - s * y_body
    r_y = s * x_body + c * y_body

    force_x = robot.propulsion_force * c
    force_y = robot.propulsion_force * s

    propulsion_torque = (
        r_x * force_y - r_y * force_x
    )

    # Contact diagnostics
    normal_contact_force = np.zeros(2)
    tangential_contact_force = np.zeros(2)

    normal_contact_torque = 0.0
    tangential_contact_torque = 0.0

    contact_results = []

    for wall in walls:

        if wall not in robot.wall_contact_states:
            robot.wall_contact_states[wall] = ContactState()

        state = robot.wall_contact_states[wall]

        result = robot_wall_contact(
            robot,
            wall,
            dt,
            state,
            contact_parameters,
        )

        contact_results.append(result)

        normal_contact_force += result.normal_force
        tangential_contact_force += result.tangential_force

        normal_contact_torque += result.normal_torque
        tangential_contact_torque += result.tangential_torque

    # Total contact contributions
    total_contact_force = (
        normal_contact_force + tangential_contact_force
    )

    total_contact_torque = (
        normal_contact_torque + tangential_contact_torque
    )

    # Total force and torque
    total_force = (
        propulsion_force
        + drag_force
        + total_contact_force
    )

    total_torque = (
        propulsion_torque
        + rotational_drag_torque
        + total_contact_torque
    )

    # Translational dynamics
    acceleration = total_force / robot.mass

    robot.velocity += acceleration * dt
    robot.position += robot.velocity * dt

    # Rotational dynamics
    angular_acceleration = (
        total_torque / robot.moment_of_inertia
    )

    robot.omega += angular_acceleration * dt
    robot.theta += robot.omega * dt

    # Return diagnostics calculated before integration
    return {
        "propulsion_force": propulsion_force.copy(),
        "drag_force": drag_force.copy(),
        "normal_contact_force": normal_contact_force.copy(),
        "tangential_contact_force": tangential_contact_force.copy(),
        "total_contact_force": total_contact_force.copy(),
        "total_force": total_force.copy(),

        "propulsion_torque": propulsion_torque,
        "rotational_drag_torque": rotational_drag_torque,
        "normal_contact_torque": normal_contact_torque,
        "tangential_contact_torque": tangential_contact_torque,
        "total_contact_torque": total_contact_torque,
        "total_torque": total_torque,

        "contact_results": contact_results,
    }
