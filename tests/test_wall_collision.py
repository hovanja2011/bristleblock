
import numpy as np

from bristlebots.robot import Robot
from bristlebots.geometry import Wall
from bristlebots.simulation import robot_wall_contact


# --------------------------------------------------
# Shared setup and integration
# --------------------------------------------------

WALL_POSITION = 1.0
RADIUS = 0.5
MASS = 1.0
INITIAL_VX = 20.0
INITIAL_VY = 5.0
INITIAL_OMEGA = 15.0
NORMAL_STIFFNESS = 4e5


def make_setup():
    """Create a robot and a vertical wall."""

    wall = Wall(
        start=[WALL_POSITION, -1.0],
        end=[WALL_POSITION, 5.0],
    )

    robot = Robot(
        position=[WALL_POSITION - RADIUS, 0.0],
        velocity=[INITIAL_VX, INITIAL_VY],
        theta=0.0,
        omega=INITIAL_OMEGA,
        radius=RADIUS,
        mass=MASS,
        propulsion_force=0.0,
        translational_drag=0.0,
        rotational_drag=0.0,
    )

    return robot, wall


def integrate_step(robot, wall, dt, mu=0.0, eta_n=0.0):
    """One integration step, including robot-wall contact."""

    force, torque = robot_wall_contact(
        robot,
        wall,
        NORMAL_STIFFNESS,
        eta_n,
        mu,
    )

    # Translational dynamics
    acceleration = force / robot.mass

    robot.velocity += acceleration * dt
    robot.position += robot.velocity * dt

    # Rotational dynamics
    angular_acceleration = torque / robot.moment_of_inertia

    robot.omega += angular_acceleration * dt
    robot.theta += robot.omega * dt

    return force, torque


def penetration(robot):
    """Return penetration depth into the right wall."""

    return max(
        robot.position[0] + robot.radius - WALL_POSITION,
        0.0,
    )


# --------------------------------------------------
# Test 1: Normal collision
# --------------------------------------------------

def test_normal_collision():
    robot, wall = make_setup()

    dt = 5e-6
    eta_n = 0.0
    mu = 0.0

    omega_c = np.sqrt(NORMAL_STIFFNESS / MASS)
    contact_time = np.pi / omega_c

    steps = int(contact_time / dt)

    time = [0.0]
    delta = [0.0]
    vx = [robot.velocity[0]]

    for i in range(steps):
        integrate_step(robot, wall, dt, mu, eta_n)

        time.append((i + 1) * dt)
        delta.append(penetration(robot))
        vx.append(robot.velocity[0])

    time = np.array(time)
    delta = np.array(delta)
    vx = np.array(vx)

    # Analytical solution
    delta_exact = (
        INITIAL_VX / omega_c
        * np.sin(omega_c * time)
    )

    vx_exact = INITIAL_VX * np.cos(omega_c * time)

    np.testing.assert_allclose(
        delta,
        delta_exact,
        rtol=0.02,
        atol=2e-4,
    )

    np.testing.assert_allclose(
        vx,
        vx_exact,
        rtol=0.02,
        atol=5e-2,
    )


# --------------------------------------------------
# Test 2: Rotation without friction
# --------------------------------------------------

def test_rotation_without_friction():
    robot, wall = make_setup()

    dt = 5e-5
    mu = 0.0
    eta_n = 0.0

    omega_initial = robot.omega
    theta_initial = robot.theta

    omega_c = np.sqrt(NORMAL_STIFFNESS / MASS)
    contact_time = np.pi / omega_c

    steps = int(contact_time / dt)

    for _ in range(steps):
        integrate_step(robot, wall, dt, mu, eta_n)

    expected_theta = theta_initial + omega_initial * steps * dt

    np.testing.assert_allclose(
        robot.omega,
        omega_initial,
        rtol=0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        robot.theta,
        expected_theta,
        rtol=0.0,
        atol=1e-10,
    )


# --------------------------------------------------
# Test 3: Kinetic friction
# --------------------------------------------------

def test_kinetic_friction():
    robot, wall = make_setup()

    dt = 1e-6
    mu = 0.3
    eta_n = 0.0

    inertia = robot.moment_of_inertia

    omega_c = np.sqrt(NORMAL_STIFFNESS / MASS)
    contact_time = np.pi / omega_c

    steps = int(np.ceil(contact_time / dt)) + 2

    # Initial invariant
    initial_invariant = (
        robot.omega
        - robot.mass * robot.radius / inertia * robot.velocity[1]
    )

    invariant_history = []
    friction_power_history = []

    for _ in range(steps):

        _, normal, overlap = wall.contact_geometry(
            robot.position,
            robot.radius,
        )

        if overlap > 0.0:

            # Contact point velocity
            r_contact = -robot.radius * normal

            rotational_velocity = robot.omega * np.array([
                -r_contact[1],
                r_contact[0],
            ])

            contact_velocity = robot.velocity + rotational_velocity

            # Normal force magnitude
            normal_velocity = np.dot(robot.velocity, normal)

            fn = max(
                NORMAL_STIFFNESS * overlap
                - eta_n * normal_velocity,
                0.0,
            )

            # Calculate forces and integrate
            force, _ = integrate_step(
                robot, wall, dt, mu, eta_n
            )

            # Tangential friction force
            tangential_force = force - fn * normal

            # Friction power
            friction_power = np.dot(
                tangential_force,
                contact_velocity,
            )

            friction_power_history.append(friction_power)

            # Check invariant
            invariant = (
                robot.omega
                - robot.mass * robot.radius / inertia * robot.velocity[1]
            )

            invariant_history.append(invariant)

        else:
            integrate_step(robot, wall, dt, mu, eta_n)

    invariant_history = np.array(invariant_history)
    friction_power_history = np.array(friction_power_history)

    assert len(invariant_history) > 0
    assert len(friction_power_history) > 0

    # Friction must never add mechanical energy.
    assert np.all(friction_power_history <= 1e-12)

    # Check conservation of the friction-related invariant.
    np.testing.assert_allclose(
        invariant_history,
        initial_invariant,
        rtol=0.0,
        atol=1e-8,
    )

# --------------------------------------------------
# Test 4: Time-step convergence
# --------------------------------------------------

def calculate_max_penetration(dt):
    robot, wall = make_setup()

    eta_n = 0.0
    mu = 0.0

    omega_c = np.sqrt(NORMAL_STIFFNESS / MASS)
    contact_time = np.pi / omega_c

    steps = int(np.ceil(contact_time / dt)) + 2

    max_delta = 0.0

    for _ in range(steps):
        integrate_step(robot, wall, dt, mu, eta_n)

        max_delta = max(
            max_delta,
            penetration(robot),
        )

    return max_delta


def test_time_step_convergence():
    analytical_max_penetration = (
        INITIAL_VX
        / np.sqrt(NORMAL_STIFFNESS / MASS)
    )

    time_steps = [
        1e-4,
        5e-5,
        2.5e-5,
    ]

    errors = []

    for dt in time_steps:
        numerical_max = calculate_max_penetration(dt)

        error = abs(
            numerical_max - analytical_max_penetration
        )

        errors.append(error)

    # The finest time step should give a more accurate result
    # than the coarsest one.
    assert errors[-1] < errors[0]
