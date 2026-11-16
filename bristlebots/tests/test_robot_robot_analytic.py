
import numpy as np

from bristlebots.robot import Robot
from bristlebots.contacts import ContactParameters
from bristlebots.simulation import step_robots


def analytical_solution(t, delta0, delta_dot0, k, c, mass):
    """
    Exact underdamped solution for the normal overlap.

    m_eff * delta'' + c * delta' + k * delta = 0
    """

    m_eff = mass / 2

    alpha = c / (2 * m_eff)
    omega_0 = np.sqrt(k / m_eff)
    omega_d = np.sqrt(omega_0**2 - alpha**2)

    A = delta0
    B = (delta_dot0 + alpha * delta0) / omega_d

    exp_factor = np.exp(-alpha * t)

    cos_term = A * np.cos(omega_d * t)
    sin_term = B * np.sin(omega_d * t)

    delta = exp_factor * (cos_term + sin_term)

    delta_dot = exp_factor * (
        -alpha * (cos_term + sin_term)
        - A * omega_d * np.sin(omega_d * t)
        + B * omega_d * np.cos(omega_d * t)
    )

    return delta, delta_dot


def test_robot_robot_collision_against_analytic_solution():

    # --------------------------------------------------
    # Parameters
    # --------------------------------------------------

    R = 1.0
    mass = 1.0

    k_n = 1000.0
    c_n = 2.0

    gamma_rot = 0.2

    dt = 1e-5
    total_time = 0.02

    delta0 = 0.01
    initial_speed = 1.0

    omega_a0 = 3.0
    omega_b0 = -2.0

    # --------------------------------------------------
    # Contact parameters
    # --------------------------------------------------

    parameters = ContactParameters(
        normal_stiffness=k_n,
        normal_damping=c_n,

        mu_static=0.0,
        mu_kinetic=0.0,

        tangential_stiffness=1000.0,
        tangential_damping=1.0,
    )

    # --------------------------------------------------
    # Initial conditions
    # --------------------------------------------------

    initial_distance = 2 * R - delta0

    robot_a = Robot(
        position=[-initial_distance / 2, 0.0],
        velocity=[initial_speed, 0.0],
        omega=omega_a0,
        radius=R,
        mass=mass,
        rotational_drag=gamma_rot,
    )

    robot_b = Robot(
        position=[initial_distance / 2, 0.0],
        velocity=[-initial_speed, 0.0],
        omega=omega_b0,
        radius=R,
        mass=mass,
        rotational_drag=gamma_rot,
    )

    robots = [robot_a, robot_b]

    # --------------------------------------------------
    # Numerical simulation
    # --------------------------------------------------

    n_steps = int(total_time / dt)

    for _ in range(n_steps):
        step_robots(
            robots,
            dt,
            contact_parameters=parameters,
        )

    # --------------------------------------------------
    # Analytical solution
    # --------------------------------------------------

    t = n_steps * dt

    delta_dot0 = 2 * initial_speed

    delta, delta_dot = analytical_solution(
        t,
        delta0,
        delta_dot0,
        k_n,
        c_n,
        mass,
    )

    expected_distance = 2 * R - delta

    expected_position_a = np.array([
        -expected_distance / 2,
        0.0,
    ])

    expected_position_b = np.array([
        expected_distance / 2,
        0.0,
    ])

    expected_velocity_a = np.array([
        delta_dot / 2,
        0.0,
    ])

    expected_velocity_b = np.array([
        -delta_dot / 2,
        0.0,
    ])

    # Rotational damping solution
    I = 0.5 * mass * R**2
    rotational_decay = np.exp(-gamma_rot * t / I)

    expected_omega_a = omega_a0 * rotational_decay
    expected_omega_b = omega_b0 * rotational_decay

    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------

    np.testing.assert_allclose(
        robot_a.position,
        expected_position_a,
        atol=2e-5,
        rtol=0,
    )

    np.testing.assert_allclose(
        robot_b.position,
        expected_position_b,
        atol=2e-5,
        rtol=0,
    )

    np.testing.assert_allclose(
        robot_a.velocity,
        expected_velocity_a,
        atol=2e-4,
        rtol=0,
    )

    np.testing.assert_allclose(
        robot_b.velocity,
        expected_velocity_b,
        atol=2e-4,
        rtol=0,
    )

    np.testing.assert_allclose(
        robot_a.omega,
        expected_omega_a,
        atol=1e-6,
        rtol=0,
    )

    np.testing.assert_allclose(
        robot_b.omega,
        expected_omega_b,
        atol=1e-6,
        rtol=0,
    )
