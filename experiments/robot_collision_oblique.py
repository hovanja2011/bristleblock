
import numpy as np
import matplotlib.pyplot as plt

from bristlebots.robot import Robot
from bristlebots.contacts import ContactParameters
from bristlebots.simulation import step_robots
from bristlebots.visualization import animate_robots


# --------------------------------------------------
# Contact parameters
# --------------------------------------------------

parameters = ContactParameters(
    normal_stiffness=1000.0,
    normal_damping=0.0,
    mu_static=0.5,
    mu_kinetic=0.3,
    tangential_stiffness=1000.0,
    tangential_damping=0.0,
)


# --------------------------------------------------
# Initial conditions
# --------------------------------------------------

R = 1.0
impact_parameter = 1.2 * R

robot_a = Robot(
    position=[0.0, 0.0],
    velocity=[1.0, 0.0],
    theta=0.0,
    omega=1.0,       # начальная угловая скорость
    radius=1.0,
    mass=1.0,
)

robot_b = Robot(
    position=[5.0, 1.2],
    velocity=[-1.0, 0.0],
    theta=0.0,
    omega=-2.0,      # вращение в противоположную сторону
    radius=1.0,
    mass=1.0,
)

robots = [robot_a, robot_b]


# --------------------------------------------------
# Simulation parameters
# --------------------------------------------------

dt = 0.001
total_time = 3.5
steps = int(total_time / dt)

record_every = 5

trajectory_history = []
theta_history = []
omega_history = []


def record_state():
    trajectory_history.append(
        [robot.position.copy() for robot in robots]
    )

    theta_history.append(
        [robot.theta for robot in robots]
    )

    omega_history.append(
        [robot.omega for robot in robots]
    )


# Record initial state
record_state()


# --------------------------------------------------
# Run simulation
# --------------------------------------------------

for i in range(steps):

    step_robots(
        robots,
        dt,
        contact_parameters=parameters,
    )

    if (i + 1) % record_every == 0:
        record_state()


trajectory_history = np.asarray(trajectory_history)
theta_history = np.asarray(theta_history)
omega_history = np.asarray(omega_history)


# --------------------------------------------------
# Animate
# --------------------------------------------------

animation = animate_robots(
    robots=robots,
    trajectory_history=trajectory_history,
    theta_history=theta_history,
    omega_history=omega_history,
    box_size=(7, 4),
    interval=20,
    trail_length=200,
    show_trails=True,
    show_angular_velocity=True,
    show_time=True,
    show_grid=True,
    robot_colors=["royalblue", "tomato"],
    figsize=(9, 6),
)

plt.show()
