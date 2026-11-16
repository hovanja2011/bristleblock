import numpy as np

from bristlebots.robot import Robot
from bristlebots.geometry import Wall
from bristlebots.contacts import ContactParameters
from bristlebots.simulation import step


def test_robot_contact_with_corner():
    """Проверяет одновременное действие сил от двух перпендикулярных стен."""

    robot = Robot(
        position=[0.4, 0.4],
        velocity=[0.0, 0.0],
        radius=0.5,
        mass=1.0,
    )

    vertical_wall = Wall(
        start=[0.0, -2.0],
        end=[0.0, 2.0],
    )

    horizontal_wall = Wall(
        start=[-2.0, 0.0],
        end=[2.0, 0.0],
    )

    parameters = ContactParameters(
        normal_stiffness=4e5,
        normal_damping=0.0,
        mu_static=0.0,
        mu_kinetic=0.0,
        tangential_stiffness=1e4,
    )

    dt = 1e-6

    step(
        robot,
        dt,
        walls=[vertical_wall, horizontal_wall],
        contact_parameters=parameters,
    )

    # Both walls should push the robot away from the corner.
    assert robot.velocity[0] > 0.0
    assert robot.velocity[1] > 0.0

    # Due to the symmetry of the setup, both velocity components
    # should be equal.
    np.testing.assert_allclose(
        robot.velocity[0],
        robot.velocity[1],
        rtol=1e-10,
    )

    expected_velocity = (
        4e5 * 0.1 / 1.0 * dt
    )

    np.testing.assert_allclose(
        robot.velocity,
        [expected_velocity, expected_velocity],
        rtol=1e-12,
        atol=1e-12,
    )

def test_energy_conservation_during_wall_collision():
    """Проверяет сохранение энергии при упругом контакте со стеной."""

    mass = 1.0
    radius = 0.5
    stiffness = 4e5

    robot = Robot(
        position=[0.499, 0.0],
        velocity=[0.0, 0.0],
        radius=radius,
        mass=mass,
    )

    wall = Wall(
        start=[0.0, -2.0],
        end=[0.0, 2.0],
    )

    parameters = ContactParameters(
        normal_stiffness=stiffness,
        normal_damping=0.0,
        mu_static=0.0,
        mu_kinetic=0.0,
        tangential_stiffness=1e4,
    )

    dt = 1e-6
    steps = 5000

    def total_energy():
        _, _, overlap = wall.contact_geometry(
            robot.position, robot.radius
        )

        kinetic = (
            0.5 * mass * np.dot(robot.velocity, robot.velocity)
            + 0.5 * robot.moment_of_inertia * robot.omega**2
        )

        potential = 0.5 * stiffness * overlap**2

        return kinetic + potential

    initial_energy = total_energy()
    energy_history = [initial_energy]

    for _ in range(steps):
        step(
            robot,
            dt,
            walls=[wall],
            contact_parameters=parameters,
        )
        energy_history.append(total_energy())

    relative_error = (
        np.max(np.abs(np.array(energy_history) - initial_energy))
        / initial_energy
    )

    # The robot must actually leave the wall after the collision.
    _, _, final_overlap = wall.contact_geometry(
        robot.position, robot.radius
    )

    assert robot.velocity[0] > 0.0
    assert final_overlap == 0.0

    assert relative_error < 0.01
def test_tangential_spring_contact_matches_analytic_solution():
    mass = 1.0
    radius = 0.5
    inertia = 0.5 * mass * radius**2

    initial_vy = 1.0
    initial_omega = 0.0

    normal_stiffness = 4e5
    tangential_stiffness = 1e4
    mu_static = 0.5
    mu_kinetic = 0.3

    dt = 1e-6
    total_time = 1e-3
    steps = round(total_time / dt)

    robot = Robot(
        position=np.array([0.499, 0.0]),
        velocity=np.array([0.0, initial_vy]),
        theta=0.0,
        omega=initial_omega,
        radius=radius,
        mass=mass,
        propulsion_force=0.0,
        translational_drag=0.0,
        rotational_drag=0.0,
    )

    wall = Wall(
        start=np.array([0.0, -2.0]),
        end=np.array([0.0, 2.0]),
    )

    parameters = ContactParameters(
        normal_stiffness=normal_stiffness,
        normal_damping=0.0,
        mu_static=mu_static,
        mu_kinetic=mu_kinetic,
        tangential_stiffness=tangential_stiffness,
        tangential_damping=0.0,
    )

    initial_overlap = radius - 0.499
    normal_force = normal_stiffness * initial_overlap
    static_limit = mu_static * normal_force

    max_tangential_force = (
        initial_vy
        * np.sqrt(
            tangential_stiffness
            / (1.0 / mass + radius**2 / inertia)
        )
    )

    assert max_tangential_force < static_limit

    for _ in range(steps):
        step(
            robot,
            dt,
            walls=[wall],
            contact_parameters=parameters,
        )

    a = 1.0 / mass + radius**2 / inertia
    omega_n = np.sqrt(tangential_stiffness * a)

    expected_vy = initial_vy - (
        initial_vy / (mass * a)
    ) * (1.0 - np.cos(omega_n * total_time))

    expected_omega = (
        radius * initial_vy / (inertia * a)
    ) * (1.0 - np.cos(omega_n * total_time))

    assert np.isclose(
        robot.velocity[1],
        expected_vy,
        rtol=2e-3,
        atol=1e-8,
    )

    assert np.isclose(
        robot.omega,
        expected_omega,
        rtol=2e-3,
        atol=1e-8,
    )

    # The robot must remain in contact with the wall.
    final_overlap = radius - robot.position[0]
    assert final_overlap > 0.0

    # The tangential spring must remain in the sticking regime.
    contact_state = robot.wall_contact_states[wall]
    tangential_force = (
        tangential_stiffness * abs(contact_state.tangential_displacement)
    )
    normal_force = normal_stiffness * final_overlap

    assert tangential_force < mu_static * normal_force


def test_wall_order_does_not_change_result():
    """Результат не должен зависеть от порядка стен."""

    def make_robot():
        return Robot(
            position=[0.4, 0.4],
            velocity=[0.0, 0.0],
            radius=0.5,
            mass=1.0,
            propulsion_force=0.0,
            translational_drag=0.0,
            rotational_drag=0.0,
        )

    vertical_wall = Wall(
        start=[0.0, -2.0],
        end=[0.0, 2.0],
    )

    horizontal_wall = Wall(
        start=[-2.0, 0.0],
        end=[2.0, 0.0],
    )

    parameters = ContactParameters(
        normal_stiffness=4e5,
        normal_damping=0.0,
        mu_static=0.0,
        mu_kinetic=0.0,
        tangential_stiffness=1e4,
    )

    dt = 1e-6

    robot_a = make_robot()
    robot_b = make_robot()

    step(
        robot_a,
        dt,
        walls=[vertical_wall, horizontal_wall],
        contact_parameters=parameters,
    )

    step(
        robot_b,
        dt,
        walls=[horizontal_wall, vertical_wall],
        contact_parameters=parameters,
    )

    np.testing.assert_allclose(
        robot_a.position,
        robot_b.position,
        rtol=0.0,
        atol=1e-14,
    )

    np.testing.assert_allclose(
        robot_a.velocity,
        robot_b.velocity,
        rtol=0.0,
        atol=1e-14,
    )

    np.testing.assert_allclose(
        robot_a.omega,
        robot_b.omega,
        rtol=0.0,
        atol=1e-14,
    )