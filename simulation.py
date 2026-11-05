
import numpy as np


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


def step(robot, dt):
    # Translational dynamics
    force = compute_forces(robot)

    acceleration = force / robot.mass

    robot.velocity += acceleration * dt
    robot.position += robot.velocity * dt

    # Rotational dynamics
    torque = compute_torque(robot)

    angular_acceleration = (
        torque / robot.moment_of_inertia
    )

    robot.omega += angular_acceleration * dt
    robot.theta += robot.omega * dt
