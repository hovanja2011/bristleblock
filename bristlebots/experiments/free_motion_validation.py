"""Quantitative free-motion benchmark against the linear-drag solution.

Run from the repository root:
    python bristlebots/experiments/free_motion_validation.py

The experiment reports max-in-time position/velocity errors and observed
orders for a fixed physical setup while the time step is refined.
"""
import numpy as np

from bristlebots.robot import Robot
from bristlebots.simulation import step


MASS = 1.4
DRAG = 0.6
PROPULSION_FORCE = 0.8
INITIAL_POSITION = np.array([-0.3, 0.2])
INITIAL_VELOCITY = np.array([0.25, -0.15])
FINAL_TIME = 2.0


def exact_state(time, x0, v0, force):
    terminal_speed = force / DRAG
    relaxation_time = MASS / DRAG
    decay = np.exp(-time / relaxation_time)
    velocity = terminal_speed + (v0 - terminal_speed) * decay
    position = (
        x0
        + terminal_speed * time
        + (v0 - terminal_speed) * relaxation_time * (1.0 - decay)
    )
    return position, velocity


def run(dt):
    robot = Robot(
        position=INITIAL_POSITION,
        velocity=INITIAL_VELOCITY,
        theta=0.0,
        radius=0.2,
        mass=MASS,
        propulsion_force=PROPULSION_FORCE,
        translational_drag=DRAG,
    )
    n_steps = round(FINAL_TIME / dt)
    position_errors = []
    velocity_errors = []

    for index in range(1, n_steps + 1):
        step(robot, dt)
        time = index * dt
        exact_x, exact_vx = exact_state(
            time, INITIAL_POSITION[0], INITIAL_VELOCITY[0], PROPULSION_FORCE
        )
        exact_y, exact_vy = exact_state(
            time, INITIAL_POSITION[1], INITIAL_VELOCITY[1], 0.0
        )
        position_errors.append(
            np.linalg.norm(robot.position - np.array([exact_x, exact_y]))
        )
        velocity_errors.append(
            np.linalg.norm(robot.velocity - np.array([exact_vx, exact_vy]))
        )

    return max(position_errors), max(velocity_errors)


def main():
    time_steps = (0.04, 0.02, 0.01, 0.005)
    results = [run(dt) for dt in time_steps]
    print("dt          max position error   order    max velocity error   order")
    previous = None
    for dt, (position_error, velocity_error) in zip(time_steps, results):
        if previous is None:
            position_order = velocity_order = float("nan")
        else:
            position_order = np.log2(previous[0] / position_error)
            velocity_order = np.log2(previous[1] / velocity_error)
        print(
            f"{dt:<11.5g} {position_error:<20.8e} "
            f"{position_order:<8.4f} {velocity_error:<20.8e} "
            f"{velocity_order:.4f}"
        )
        previous = (position_error, velocity_error)


if __name__ == "__main__":
    main()
