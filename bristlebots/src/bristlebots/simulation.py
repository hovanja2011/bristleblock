"""Time integration of bristlebots with propulsion, drag and contact forces."""
import numpy as np

from bristlebots.contacts import (
    ContactState,
    robot_wall_contact,
    robot_robot_contact,
)

def step(robot, dt, walls=None, contact_parameters=None):
    """
    Advance robot dynamics by one time step.
    Return force and torque diagnostics calculated before integration.
    """

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Time step must be positive.")

    if walls is None:
        walls = []

    if walls and contact_parameters is None:
        raise ValueError(
            "Contact parameters must be provided when walls are present."
        )

    # Free-motion forces
    propulsion_force = robot.propulsion_force_vector

    drag_force = (
        -robot.translational_drag * robot.velocity
    )

    # Free-motion torques
    rotational_drag_torque = (
        -robot.rotational_drag * robot.omega
    )
    active_torque = robot.active_torque

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
        active_torque
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

        "active_torque": robot.active_torque,
        "rotational_drag_torque": rotational_drag_torque,
        "normal_contact_torque": normal_contact_torque,
        "tangential_contact_torque": tangential_contact_torque,
        "total_contact_torque": total_contact_torque,
        "total_torque": total_torque,

        "contact_results": contact_results,
    }

def step_robots(
    robots,
    dt,
    walls=None,
    contact_parameters=None,
    external_force=None,
):
    """
    Advance a system of robots by one time step.

    All forces are calculated from the current states before
    any robot is updated.

    Return a list of force and torque diagnostics, in the
    same order as the input robots.
    """

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Time step must be positive.")

    robots = list(robots)

    if len({id(robot) for robot in robots}) != len(robots):
        raise ValueError("The robots list must not contain duplicates.")

    if walls is None:
        walls = []

    if (walls or len(robots) > 1) and contact_parameters is None:
        raise ValueError(
            "Contact parameters must be provided when contacts are enabled."
        )

    if external_force is None:
        external_force = np.zeros(2)
    else:
        external_force = np.asarray(
            external_force,
            dtype=float,
        )

        if external_force.shape != (2,):
            raise ValueError(
                "external_force must be a 2D vector."
            )

        if not np.all(np.isfinite(external_force)):
            raise ValueError(
                "external_force must contain finite values."
            )
    # --------------------------------------------------
    # Initialize force and torque accumulators
    # --------------------------------------------------

    forces = {
        robot: np.zeros(2)
        for robot in robots
    }

    torques = {
        robot: 0.0
        for robot in robots
    }

    diagnostics = {}

    # --------------------------------------------------
    # Free-motion forces and wall contacts
    # --------------------------------------------------

    for robot in robots:

        propulsion_force = robot.propulsion_force_vector

        drag_force = (
            -robot.translational_drag * robot.velocity
        )

        rotational_drag_torque = (
            -robot.rotational_drag * robot.omega
        )

        active_torque = robot.active_torque

        # Wall-contact contributions
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

        total_wall_contact_force = (
            normal_contact_force + tangential_contact_force
        )

        total_wall_contact_torque = (
            normal_contact_torque + tangential_contact_torque
        )

        forces[robot] += (
            propulsion_force
            + drag_force
            + external_force
            + total_wall_contact_force
        )

        torques[robot] += (
            active_torque
            + rotational_drag_torque
            + total_wall_contact_torque
        )

        diagnostics[robot] = {
            "propulsion_force": propulsion_force.copy(),
            "drag_force": drag_force.copy(),

            "normal_contact_force": normal_contact_force.copy(),
            "tangential_contact_force": tangential_contact_force.copy(),
            "total_contact_force": total_wall_contact_force.copy(),

            "active_torque": robot.active_torque,
            "rotational_drag_torque": rotational_drag_torque,
            "normal_contact_torque": normal_contact_torque,
            "tangential_contact_torque": tangential_contact_torque,
            "total_contact_torque": total_wall_contact_torque,

            "contact_results": contact_results,
            "robot_contact_results": [],

            "external_force": external_force.copy(),
        }

    # --------------------------------------------------
    # Robot-robot contacts
    # --------------------------------------------------

    for i in range(len(robots)):
        for j in range(i + 1, len(robots)):

            robot_a = robots[i]
            robot_b = robots[j]

            if robot_b not in robot_a.robot_contact_states:
                robot_a.robot_contact_states[robot_b] = ContactState()

            state = robot_a.robot_contact_states[robot_b]

            result = robot_robot_contact(
                robot_a,
                robot_b,
                dt,
                state,
                contact_parameters,
            )

            # Newton's third law
            forces[robot_a] += result.force_on_first
            forces[robot_b] += result.force_on_second

            torques[robot_a] += result.torque_on_first
            torques[robot_b] += result.torque_on_second

            # Keep the result accessible from both robots
            diagnostics[robot_a]["robot_contact_results"].append(
                (robot_b, result)
            )

            diagnostics[robot_b]["robot_contact_results"].append(
                (robot_a, result)
            )
    # --------------------------------------------------
    # Store total force and torque diagnostics
    # --------------------------------------------------

    for robot in robots:

        diag = diagnostics[robot]

        robot_robot_force = (
            forces[robot]
            - diag["propulsion_force"]
            - diag["drag_force"]
            - external_force
            - diag["total_contact_force"]
        )

        robot_robot_torque = (
            torques[robot]
            - diag["active_torque"]
            - diag["rotational_drag_torque"]
            - diag["total_contact_torque"]
        )

        diag["robot_robot_contact_force"] = robot_robot_force.copy()
        diag["robot_robot_contact_torque"] = robot_robot_torque

        diag["total_contact_force"] += robot_robot_force
        diag["total_contact_torque"] += robot_robot_torque

        diag["total_force"] = forces[robot].copy()
        diag["total_torque"] = torques[robot]

    # --------------------------------------------------
    # Synchronous integration
    # --------------------------------------------------

    # First update velocities and angular velocities
    for robot in robots:

        acceleration = forces[robot] / robot.mass

        angular_acceleration = (
            torques[robot] / robot.moment_of_inertia
        )

        robot.velocity += acceleration * dt
        robot.omega += angular_acceleration * dt

    # Then update positions and orientations
    for robot in robots:

        robot.position += robot.velocity * dt
        robot.theta += robot.omega * dt

    return [
        diagnostics[robot]
        for robot in robots
    ]