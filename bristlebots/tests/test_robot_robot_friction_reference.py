
import numpy as np
from scipy.integrate import solve_ivp

from bristlebots.robot import Robot
from bristlebots.contacts import (
    ContactParameters,
    ContactState,
    robot_robot_contact,
)
from bristlebots.simulation import step_robots


# --------------------------------------------------
# Physical parameters
# --------------------------------------------------

R = 1.0
M = 1.0
I = 0.5 * M * R**2

GAMMA_T = 0.1
GAMMA_R = 0.2

KN = 1000.0
CN = 2.0
KT = 500.0
CT = 0.5

TOTAL_TIME = 0.01

DT_VALUES = [1e-5, 5e-6]


parameters = ContactParameters(
    normal_stiffness=KN,
    normal_damping=CN,
    mu_static=100.0,
    mu_kinetic=100.0,
    tangential_stiffness=KT,
    tangential_damping=CT,
)


# --------------------------------------------------
# Initial conditions
# --------------------------------------------------

def make_robots():

    robot_a = Robot(
        position=(0.0, 0.0),
        velocity=(0.5, 0.0),
        theta=0.0,
        omega=0.8,
        radius=R,
        mass=M,
        translational_drag=GAMMA_T,
        rotational_drag=GAMMA_R,
    )

    robot_b = Robot(
        position=(1.99, 0.15),
        velocity=(-0.5, 0.0),
        theta=0.0,
        omega=-0.4,
        radius=R,
        mass=M,
        translational_drag=GAMMA_T,
        rotational_drag=GAMMA_R,
    )

    return robot_a, robot_b


# --------------------------------------------------
# Independent continuous-time reference model
# --------------------------------------------------

def reference_rhs(t, y):

    # Robot A
    pos_a = y[0:2]
    vel_a = y[2:4]
    theta_a = y[4]
    omega_a = y[5]

    # Robot B
    pos_b = y[6:8]
    vel_b = y[8:10]
    theta_b = y[10]
    omega_b = y[11]

    # Tangential contact displacement
    xi = y[12]

    displacement = pos_b - pos_a
    distance = np.linalg.norm(displacement)

    overlap = 2.0 * R - distance

    if overlap <= 0:
        raise RuntimeError("Reference solution lost contact.")

    normal = displacement / distance
    tangent = np.array([-normal[1], normal[0]])

    normal_velocity = np.dot(vel_b - vel_a, normal)

    fn = max(
        0.0,
        KN * overlap - CN * normal_velocity,
    )

    # A common midpoint contact point is used for the compliant overlap.
    contact_arm = 0.5 * distance
    velocity_contact_a = vel_a + omega_a * contact_arm * tangent
    velocity_contact_b = vel_b - omega_b * contact_arm * tangent

    vt = np.dot(
        velocity_contact_a - velocity_contact_b,
        tangent,
    )

    # Sticking friction law
    ft = -KT * xi - CT * vt

    if abs(ft) > parameters.mu_static * fn:
        raise RuntimeError(
            "Reference solution left the sticking regime."
        )

    force_a = -fn * normal + ft * tangent
    force_b = -force_a

    torque_a = contact_arm * ft
    torque_b = contact_arm * ft

    # Equations of motion
    acceleration_a = (
        force_a - GAMMA_T * vel_a
    ) / M

    acceleration_b = (
        force_b - GAMMA_T * vel_b
    ) / M

    angular_acceleration_a = (
        torque_a - GAMMA_R * omega_a
    ) / I

    angular_acceleration_b = (
        torque_b - GAMMA_R * omega_b
    ) / I

    return np.array([
        vel_a[0],
        vel_a[1],
        acceleration_a[0],
        acceleration_a[1],
        omega_a,
        angular_acceleration_a,

        vel_b[0],
        vel_b[1],
        acceleration_b[0],
        acceleration_b[1],
        omega_b,
        angular_acceleration_b,

        vt,
    ])


def solve_reference():

    robot_a, robot_b = make_robots()

    y0 = np.array([
        *robot_a.position,
        *robot_a.velocity,
        robot_a.theta,
        robot_a.omega,

        *robot_b.position,
        *robot_b.velocity,
        robot_b.theta,
        robot_b.omega,

        0.0,
    ])

    solution = solve_ivp(
        reference_rhs,
        (0.0, TOTAL_TIME),
        y0,
        method="DOP853",
        rtol=1e-12,
        atol=1e-14,
    )

    assert solution.success

    return solution.y[:, -1]


# --------------------------------------------------
# Discrete simulation
# --------------------------------------------------

def run_simulation(dt):

    robot_a, robot_b = make_robots()

    steps = round(TOTAL_TIME / dt)

    assert np.isclose(steps * dt, TOTAL_TIME)

    sticking_steps = 0
    nonzero_torque_steps = 0

    for _ in range(steps):

        diagnostics = step_robots(
            [robot_a, robot_b],
            dt,
            contact_parameters=parameters,
        )

        contact_result = (
            diagnostics[0]["robot_contact_results"][0][1]
        )

        assert contact_result.overlap > 0

        assert contact_result.mode == "sticking"

        sticking_steps += 1

        if abs(contact_result.tangential_force_magnitude) > 1e-12:
            nonzero_torque_steps += 1

    assert sticking_steps == steps
    assert nonzero_torque_steps > 0

    xi = robot_a.robot_contact_states[
        robot_b
    ].tangential_displacement

    final_state = np.array([
        *robot_a.position,
        *robot_a.velocity,
        robot_a.theta,
        robot_a.omega,

        *robot_b.position,
        *robot_b.velocity,
        robot_b.theta,
        robot_b.omega,

        xi,
    ])

    return final_state


# --------------------------------------------------
# Test
# --------------------------------------------------

def test_robot_robot_friction_against_reference():

    reference = solve_reference()

    errors = []

    for dt in DT_VALUES:

        numerical = run_simulation(dt)

        error = np.linalg.norm(
            numerical - reference
        )

        errors.append(error)

        print(
            f"\ndt = {dt:.1e}, "
            f"error = {error:.8e}"
        )

    # Smaller time step should give a smaller error
    assert errors[1] < errors[0]

    # The finer simulation should be reasonably close
    assert errors[1] < 1e-3

    # Friction must change the angular velocities
    # relative to free rotational damping.
    omega_a_free = 0.8 * np.exp(
        -GAMMA_R * TOTAL_TIME / I
    )

    omega_b_free = -0.4 * np.exp(
        -GAMMA_R * TOTAL_TIME / I
    )

    assert not np.isclose(
        reference[5],
        omega_a_free,
        atol=1e-8,
    )

    assert not np.isclose(
        reference[11],
        omega_b_free,
        atol=1e-8,
    )


def test_robot_robot_sliding_friction():

    robot_a = Robot(
        position=(0.0, 0.0),
        velocity=(0.0, 0.0),
        radius=1.0,
        mass=1.0,
    )

    robot_b = Robot(
        position=(1.99, 0.0),
        velocity=(0.0, 0.0),
        radius=1.0,
        mass=1.0,
    )

    params = ContactParameters(
        normal_stiffness=1000.0,
        normal_damping=0.0,
        mu_static=0.2,
        mu_kinetic=0.1,
        tangential_stiffness=500.0,
        tangential_damping=0.0,
    )

    # Initial tangential deformation is large enough
    # to exceed the static friction limit.
    state = ContactState(
        tangential_displacement=0.01,
        previous_tangent=np.array([0.0, 1.0]),
    )

    result = robot_robot_contact(
        robot_a,
        robot_b,
        dt=1e-5,
        state=state,
        parameters=params,
    )

    normal_force = 1000.0 * 0.01

    kinetic_limit = (
        params.mu_kinetic * normal_force
    )

    assert result.mode == "sliding"

    # Coulomb kinetic friction limit
    np.testing.assert_allclose(
        result.tangential_force_magnitude,
        kinetic_limit,
        rtol=1e-12,
        atol=1e-12,
    )

    # Forces must obey Newton's third law
    np.testing.assert_allclose(
        result.force_on_first,
        -result.force_on_second,
    )

    # Friction must produce contact torques
    assert abs(result.torque_on_first) > 0
    assert abs(result.torque_on_second) > 0

    # Both torques have the same sign in this model
    assert (
        np.sign(result.torque_on_first)
        == np.sign(result.torque_on_second)
    )


def test_robot_robot_contact_break_resets_history():

    robot_a = Robot(
        position=(0.0, 0.0),
        velocity=(0.0, 0.0),
        radius=1.0,
        mass=1.0,
    )

    robot_b = Robot(
        position=(1.99, 0.0),
        velocity=(0.0, 0.0),
        radius=1.0,
        mass=1.0,
    )

    params = ContactParameters(
        normal_stiffness=1000.0,
        normal_damping=0.0,
        mu_static=1.0,
        mu_kinetic=0.5,
        tangential_stiffness=500.0,
        tangential_damping=0.0,
    )

    state = ContactState()

    # First call: contact exists
    result_contact = robot_robot_contact(
        robot_a,
        robot_b,
        dt=1e-5,
        state=state,
        parameters=params,
    )

    assert result_contact.mode == "sticking"
    assert result_contact.overlap > 0

    # Create a nonzero tangential history
    assert state.previous_tangent is not None

    state.tangential_displacement = 0.005

    # Move robot B far enough to break contact
    robot_b.position = np.array([2.1, 0.0])

    result_no_contact = robot_robot_contact(
        robot_a,
        robot_b,
        dt=1e-5,
        state=state,
        parameters=params,
    )

    assert result_no_contact.mode == "no_contact"

    np.testing.assert_allclose(
        result_no_contact.force_on_first,
        np.zeros(2),
    )

    np.testing.assert_allclose(
        result_no_contact.force_on_second,
        np.zeros(2),
    )

    assert result_no_contact.torque_on_first == 0.0
    assert result_no_contact.torque_on_second == 0.0

    # Contact history must be completely reset
    assert state.tangential_displacement == 0.0
    assert state.previous_tangent is None
