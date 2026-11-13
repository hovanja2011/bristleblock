
import numpy as np

from bristlebots.robot import Robot
from bristlebots.geometry import Wall
from bristlebots.contacts import ContactParameters, ContactState, robot_wall_contact
from bristlebots.simulation import step

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

    contact_parameters = ContactParameters(
        normal_stiffness=NORMAL_STIFFNESS,
        normal_damping=eta_n,
        mu_static=mu,
        mu_kinetic=mu,
        tangential_stiffness=1e4,
        tangential_damping=0.0,
    )

    step(
        robot,
        dt,
        walls=[wall],
        contact_parameters=contact_parameters,
    )


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

    for _ in range(steps):

        _, _, overlap = wall.contact_geometry(
            robot.position,
            robot.radius,
        )

        integrate_step(robot, wall, dt, mu, eta_n)

        if overlap > 0.0:
            invariant = (
                robot.omega
                - robot.mass * robot.radius / inertia * robot.velocity[1]
            )

            invariant_history.append(invariant)

    invariant_history = np.array(invariant_history)

    assert len(invariant_history) > 0

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
    assert errors[0] > errors[1] > errors[2]


def make_contact_parameters(
    mu_static,
    mu_kinetic,
    tangential_stiffness=1e4,
):
    return ContactParameters(
        normal_stiffness=NORMAL_STIFFNESS,
        normal_damping=0.0,
        mu_static=mu_static,
        mu_kinetic=mu_kinetic,
        tangential_stiffness=tangential_stiffness,
        tangential_damping=0.0,
    )

def test_static_friction():
    """Проверяет, что касательная сила не превышает предел статического трения."""
    robot, wall = make_setup()

    robot.position[0] += 1e-4  # Create a small overlap with the wall

    dt = 1e-6
    parameters = make_contact_parameters(
        mu_static=0.5,
        mu_kinetic=0.3,
    )

    state = ContactState()

    result = robot_wall_contact(
        robot, wall, 1e-6, state, parameters
    )

    force = result.total_force

    # The contact point initially has tangential velocity.
    r_contact = np.array([robot.radius, 0.0])
    contact_velocity = robot.velocity + robot.omega * np.array([
        -r_contact[1],
        r_contact[0],
    ])

    tangent = np.array([0.0, -1.0])
    vt = np.dot(contact_velocity, tangent)

    assert vt != 0.0

    # The tangential force must remain below the static limit.
    fn = np.dot(force, np.array([-1.0, 0.0]))
    ft = np.dot(force, tangent)

    assert abs(ft) <= parameters.mu_static * fn
    assert np.linalg.norm(force) > 0.0


def test_kinetic_friction_limit():
    """Проверяет, что при превышении предела статического трения сила ограничивается кинетическим пределом."""

    robot, wall = make_setup()
    robot.position[0] += 1e-4  # Ненулевое перекрытие!

    dt = 1e-6
    parameters = make_contact_parameters(
        mu_static=0.3,
        mu_kinetic=0.2,
        tangential_stiffness=1e6,
    )

    state = ContactState()

    # Artificially create a large tangential spring displacement.
    state.tangential_displacement = 1.0

    result = robot_wall_contact(
        robot, wall, 1e-6, state, parameters
    )

    force = result.total_force
    torque = result.total_torque

    normal = np.array([-1.0, 0.0])
    tangent = np.array([0.0, -1.0])

    fn = np.dot(force, normal)
    ft = np.dot(force, tangent)

    expected_fn = NORMAL_STIFFNESS * 1e-4

    np.testing.assert_allclose(fn, expected_fn, rtol=1e-12)

    np.testing.assert_allclose(
        abs(ft),
        parameters.mu_kinetic * expected_fn,
        rtol=1e-10,
    )

    expected_torque_magnitude = RADIUS * abs(ft)

    np.testing.assert_allclose(
        abs(torque),
        expected_torque_magnitude,
        rtol=1e-12,
        atol=1e-12,
    )

def test_stick_slip_transition():
    """Проверяет переключение алгоритма от статического к кинетическому трению."""
    robot, wall = make_setup()

    robot.position[0] += 1e-4  # Create a small overlap with the wall

    dt = 1e-6
    parameters = make_contact_parameters(
        mu_static=0.5,
        mu_kinetic=0.3,
        tangential_stiffness=1e4,
    )

    state = ContactState()

    # Initially, tangential displacement is zero.
    assert state.tangential_displacement == 0.0

    # First contact calculation.
    result = robot_wall_contact(
        robot, wall, 1e-6, state, parameters
    )

    force = result.total_force

    normal = np.array([-1.0, 0.0])
    tangent = np.array([0.0, -1.0])

    fn = np.dot(force, normal)
    ft = np.dot(force, tangent)

    assert abs(ft) < parameters.mu_static * fn

    # Force a large accumulated tangential displacement.
    state.tangential_displacement = 1.0

    result = robot_wall_contact(
        robot, wall, 1e-6, state, parameters
    )

    force = result.total_force

    fn = np.dot(force, normal)
    ft = np.dot(force, tangent)

    np.testing.assert_allclose(
        abs(ft),
        parameters.mu_kinetic * fn,
        rtol=1e-10,
    )

def test_normal_contact_force_with_damping():
    """Проверяет нормальную силу по аналитической формуле."""

    robot, wall = make_setup()

    overlap = 1e-4
    damping = 10.0

    robot.position[0] += overlap

    parameters = make_contact_parameters(
        mu_static=0.0,
        mu_kinetic=0.0,
    )

    parameters.normal_damping = damping

    state = ContactState()

    result = robot_wall_contact(
        robot, wall, 1e-6, state, parameters
    )

    force = result.total_force
    torque = result.total_torque

    expected_fn = (
        NORMAL_STIFFNESS * overlap
        + damping * INITIAL_VX
    )

    expected_force = np.array([-expected_fn, 0.0])

    np.testing.assert_allclose(
        force,
        expected_force,
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        torque,
        0.0,
        atol=1e-12,
    )

def test_damped_normal_collision():
    """Сравнивает численное столкновение с аналитическим решением."""

    robot, wall = make_setup()

    dt = 2e-7
    eta_n = 40.0

    parameters = ContactParameters(
        normal_stiffness=NORMAL_STIFFNESS,
        normal_damping=eta_n,
        mu_static=0.0,
        mu_kinetic=0.0,
        tangential_stiffness=1e4,
        tangential_damping=0.0,
    )

    gamma = eta_n / (2.0 * MASS)
    omega_d = np.sqrt(
        NORMAL_STIFFNESS / MASS - gamma**2
    )

    contact_time = np.pi / omega_d
    steps = int(contact_time / dt)

    time = np.arange(steps + 1) * dt

    delta_numerical = [penetration(robot)]
    vx_numerical = [robot.velocity[0]]

    for _ in range(steps):
        step(
            robot,
            dt,
            walls=[wall],
            contact_parameters=parameters,
        )

        delta_numerical.append(penetration(robot))
        vx_numerical.append(robot.velocity[0])

    delta_numerical = np.array(delta_numerical)
    vx_numerical = np.array(vx_numerical)

    decay = np.exp(-gamma * time)

    delta_exact = (
        INITIAL_VX / omega_d
        * decay
        * np.sin(omega_d * time)
    )

    vx_exact = (
        INITIAL_VX
        * decay
        * (
            np.cos(omega_d * time)
            - gamma / omega_d * np.sin(omega_d * time)
        )
    )

    np.testing.assert_allclose(
        delta_numerical,
        delta_exact,
        rtol=5e-3,
        atol=2.5e-6,
    )

    np.testing.assert_allclose(
        vx_numerical,
        vx_exact,
        rtol=5e-3,
        atol=2e-3,
    )