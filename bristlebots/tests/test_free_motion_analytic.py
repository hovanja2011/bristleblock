"""Analytical reference test for free propulsion with linear drag."""
import numpy as np

from bristlebots.robot import Robot
from bristlebots.simulation import step


def _exact_state(time, *, x0, v0, force, mass, drag):
    """Exact 1D solution of m dv/dt = force - drag*v."""
    terminal_speed = force / drag
    relaxation_time = mass / drag
    decay = np.exp(-time / relaxation_time)
    velocity = terminal_speed + (v0 - terminal_speed) * decay
    position = (
        x0
        + terminal_speed * time
        + (v0 - terminal_speed) * relaxation_time * (1.0 - decay)
    )
    return position, velocity


def _run(dt, final_time=2.0):
    mass = 1.4
    drag = 0.6
    force = 0.8
    initial_position = np.array([-0.3, 0.2])
    initial_velocity = np.array([0.25, -0.15])

    robot = Robot(
        position=initial_position,
        velocity=initial_velocity,
        theta=0.0,
        radius=0.2,
        mass=mass,
        propulsion_force=force,
        translational_drag=drag,
    )
    n_steps = round(final_time / dt)
    velocity_errors = []
    position_errors = []

    for index in range(1, n_steps + 1):
        step(robot, dt)
        time = index * dt
        exact_x, exact_vx = _exact_state(
            time,
            x0=initial_position[0],
            v0=initial_velocity[0],
            force=force,
            mass=mass,
            drag=drag,
        )
        exact_y, exact_vy = _exact_state(
            time,
            x0=initial_position[1],
            v0=initial_velocity[1],
            force=0.0,
            mass=mass,
            drag=drag,
        )
        exact_position = np.array([exact_x, exact_y])
        exact_velocity = np.array([exact_vx, exact_vy])
        position_errors.append(np.linalg.norm(robot.position - exact_position))
        velocity_errors.append(np.linalg.norm(robot.velocity - exact_velocity))

    return max(position_errors), max(velocity_errors)


def test_free_motion_converges_to_analytic_position_and_velocity():
    """The semi-implicit Euler trajectory should converge at first order."""
    time_steps = (0.04, 0.02, 0.01, 0.005)
    position_errors = []
    velocity_errors = []

    for dt in time_steps:
        position_error, velocity_error = _run(dt)
        position_errors.append(position_error)
        velocity_errors.append(velocity_error)

    assert np.all(np.diff(position_errors) < 0.0)
    assert np.all(np.diff(velocity_errors) < 0.0)

    position_orders = np.log2(
        np.asarray(position_errors[:-1]) / np.asarray(position_errors[1:])
    )
    velocity_orders = np.log2(
        np.asarray(velocity_errors[:-1]) / np.asarray(velocity_errors[1:])
    )
    assert np.all((position_orders > 0.8) & (position_orders < 1.2))
    assert np.all((velocity_orders > 0.8) & (velocity_orders < 1.2))
