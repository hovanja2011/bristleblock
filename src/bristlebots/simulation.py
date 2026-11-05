
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


def robot_wall_force(robot, wall, normal_stiffness, normal_damping):
    """
    Calculate normal contact force between a robot and a wall.
    Tangential friction is not included yet.
    """
    distance, normal, overlap = wall.contact_geometry(
        robot.position, robot.radius
    )

    if overlap <= 0.0:
        return np.zeros(2)

    normal_velocity = np.dot(robot.velocity, normal)

    magnitude = (
        normal_stiffness * overlap
        - normal_damping * normal_velocity
    )

    magnitude = max(magnitude, 0.0)

    return magnitude * normal

def robot_wall_contact(robot, wall, normal_stiffness,
                       normal_damping, friction_coefficient):

    distance, normal, overlap = wall.contact_geometry(
        robot.position, robot.radius
    )

    if overlap <= 0.0:
        return np.zeros(2), 0.0

    # Normal force
    normal_velocity = np.dot(robot.velocity, normal)

    fn = normal_stiffness * overlap - normal_damping * normal_velocity
    fn = max(fn, 0.0)

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

    # Kinetic friction
    tangential_force = (
        -friction_coefficient
        * fn
        * np.sign(tangential_velocity)
        * tangent
    )

    # Total force
    force = normal_force + tangential_force

    # Torque around robot center
    torque = (
        r_contact[0] * tangential_force[1]
        - r_contact[1] * tangential_force[0]
    )

    return force, torque