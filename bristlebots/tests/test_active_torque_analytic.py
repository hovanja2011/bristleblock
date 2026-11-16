
import numpy as np
import pytest

from bristlebots.robot import Robot
from bristlebots.simulation import step, step_robots


@pytest.mark.parametrize("integrator", [step, step_robots])
def test_active_torque_with_rotational_drag(integrator):
    mass = 1.0
    radius = 1.0
    inertia = 0.5 * mass * radius**2

    active_torque = 0.8
    rotational_drag = 0.25

    omega_0 = 0.3
    theta_0 = 0.2

    dt = 1e-4
    steps = 1000

    robot = Robot(
        position=(0.0, 0.0),
        velocity=(0.0, 0.0),
        theta=theta_0,
        omega=omega_0,
        radius=radius,
        mass=mass,
        propulsion_force=0.0,
        active_torque=active_torque,
        translational_drag=0.0,
        rotational_drag=rotational_drag,
    )

    for _ in range(steps):
        if integrator is step:
            integrator(robot, dt)
        else:
            integrator([robot], dt)

    # Exact discrete solution for semi-implicit Euler integration
    a = 1.0 - rotational_drag * dt / inertia
    omega_inf = active_torque / rotational_drag

    expected_omega = (
        omega_inf + (omega_0 - omega_inf) * a**steps
    )

    expected_theta = (
        theta_0
        + dt * (
            steps * omega_inf
            + (omega_0 - omega_inf)
            * a
            * (1.0 - a**steps)
            / (1.0 - a)
        )
    )

    assert robot.omega == pytest.approx(expected_omega, abs=1e-10)
    assert robot.theta == pytest.approx(expected_theta, abs=1e-10)

    # Active torque must not produce translational motion.
    np.testing.assert_allclose(
        robot.position,
        (0.0, 0.0),
        atol=1e-12,
    )

    np.testing.assert_allclose(
        robot.velocity,
        (0.0, 0.0),
        atol=1e-12,
    )
