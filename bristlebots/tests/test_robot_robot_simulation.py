
import numpy as np

from bristlebots.robot import Robot
from bristlebots.contacts import ContactParameters
from bristlebots.simulation import step_robots


def test_head_on_robot_collision():

    parameters = ContactParameters(
        normal_stiffness=1000.0,
        normal_damping=0.0,
        mu_static=0.0,
        mu_kinetic=0.0,
        tangential_stiffness=1000.0,
        tangential_damping=0.0,
    )

    robot_a = Robot(
        position=[0.0, 0.0],
        velocity=[1.0, 0.0],
        radius=1.0,
        mass=1.0,
    )

    robot_b = Robot(
        position=[1.9, 0.0],
        velocity=[-1.0, 0.0],
        radius=1.0,
        mass=1.0,
    )

    dt = 0.001

    print("\n--- INITIAL STATE ---")
    print("Robot A position:", robot_a.position)
    print("Robot A velocity:", robot_a.velocity)
    print("Robot B position:", robot_b.position)
    print("Robot B velocity:", robot_b.velocity)

    distance = np.linalg.norm(
        robot_b.position - robot_a.position
    )
    overlap = robot_a.radius + robot_b.radius - distance

    print("\n--- INITIAL CONTACT GEOMETRY ---")
    print("Distance between centers:", distance)
    print("Overlap:", overlap)

    results = step_robots(
        [robot_a, robot_b],
        dt,
        contact_parameters=parameters,
    )

    result_a, result_b = results

    force_a = result_a["robot_robot_contact_force"]
    force_b = result_b["robot_robot_contact_force"]

    print("\n--- CONTACT FORCES ---")
    print("Force on A:", force_a)
    print("Force on B:", force_b)
    print("Sum of contact forces:", force_a + force_b)

    print("\n--- CONTACT TORQUES ---")
    print("Torque on A:", result_a["robot_robot_contact_torque"])
    print("Torque on B:", result_b["robot_robot_contact_torque"])

    print("\n--- STATE AFTER INTEGRATION ---")
    print("Robot A position:", robot_a.position)
    print("Robot A velocity:", robot_a.velocity)
    print("Robot B position:", robot_b.position)
    print("Robot B velocity:", robot_b.velocity)

    print("\n--- CHECKS ---")

    assert force_a[0] < 0.0
    assert force_b[0] > 0.0

    np.testing.assert_allclose(force_a, -force_b)

    np.testing.assert_allclose(
        force_a[1], 0.0, atol=1e-12
    )

    assert abs(result_a["robot_robot_contact_torque"]) < 1e-12
    assert abs(result_b["robot_robot_contact_torque"]) < 1e-12

    assert robot_a.velocity[0] < 1.0
    assert robot_b.velocity[0] > -1.0

    print("All checks passed.")
