
import numpy as np
import matplotlib.pyplot as plt

from robot import Robot
from simulation import step


robot = Robot(
    position=[0.0, 0.0],
    theta=0.0,
    radius=1.0,
    mass=1.0,
    propulsion_force=1.0,
    translational_drag=1.0,
)

dt = 0.001
total_time = 10.0
steps = int(total_time / dt)

times = []
speeds = []
positions = []

for i in range(steps):
    step(robot, dt)

    times.append(i * dt)
    speeds.append(np.linalg.norm(robot.velocity))
    positions.append(robot.position.copy())


times = np.array(times)
speeds = np.array(speeds)
positions = np.array(positions)

# Analytical solution
v0 = robot.free_speed
tau = robot.mass / robot.translational_drag

theoretical_speed = v0 * (
    1.0 - np.exp(-times / tau)
)

# Plot speed
plt.figure(figsize=(8, 5))

plt.plot(
    times,
    speeds,
    label="Numerical solution"
)

plt.plot(
    times,
    theoretical_speed,
    "--",
    label="Analytical solution"
)

plt.xlabel("Time")
plt.ylabel("Speed")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Plot trajectory
plt.figure(figsize=(6, 4))

plt.plot(
    positions[:, 0],
    positions[:, 1]
)

plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid()
plt.tight_layout()
plt.show()
