
import matplotlib.pyplot as plt

from bristlebots.robot import Robot
from bristlebots.simulation import step
from bristlebots.visualization import animate_robots


# Create one robot
robot = Robot(
    position=[5.0, 5.0],
    velocity=[20.0, 5.0],
    theta=0.3,
    radius=1.0,
    mass=1.0,
    propulsion_force=1.0,
    translational_drag=1.0,
)

robots = [robot]

# Simulation parameters
dt = 0.01
total_time = 20.0

steps = int(total_time / dt)

trajectory_history = []

# Run simulation
for _ in range(steps):

    for robot in robots:
        step(robot, dt)

    positions = [
        robot.position.copy()
        for robot in robots
    ]

    trajectory_history.append(positions)


# Convert to NumPy array
trajectory_history = (
    trajectory_history
)

# Animate
animation = animate_robots(
    robots,
    trajectory_history,
    box_size=(20, 20),
)

plt.show()
