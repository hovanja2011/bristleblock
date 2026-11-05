
import numpy as np
import matplotlib.pyplot as plt

from bristlebots.robot import Robot
from ..src.bristlebots.geometry import Wall
from ..src.bristlebots.simulation import robot_wall_contact
from ..src.bristlebots.visualization import animate_robots

# Vertical wall at x = 5
wall = Wall(start=[5.0, -5.0], end=[5.0, 5.0])

robot = Robot(
    position=np.array([0.0, 0.0]),
    velocity=np.array([20.0, 5.0]),
    theta=0.0,
    omega=5.0,
    radius=0.5,
    mass=1.0,
    propulsion_force=0.0,
    translational_drag=0.0,
    rotational_drag=0.0,
)

k_n = 10000.0
eta_n = 20.0

dt = 1e-4
total_time = 2.0
steps = int(total_time / dt)

mu = 0.3
moment_of_inertia = 0.5 * robot.mass * robot.radius**2

time = []
x = []
vx = []
trajectory_history = []
theta_history = []
omega_history = []

for step in range(steps):
    force, torque = robot_wall_contact(
        robot, wall, k_n, eta_n, mu
    )

    # Newton's second law
    acceleration = force / robot.mass

    # Semi-implicit Euler integration
    robot.velocity += acceleration * dt
    robot.position += robot.velocity * dt

    # Rotational motion
    angular_acceleration = torque / moment_of_inertia

    robot.omega += angular_acceleration * dt
    robot.theta += robot.omega * dt

    time.append(step * dt)
    x.append(robot.position[0])
    vx.append(robot.velocity[0])
    trajectory_history.append(np.array([robot.position.copy()]))
    theta_history.append([robot.theta])
    omega_history.append([robot.omega])

time = np.array(time)
x = np.array(x)
vx = np.array(vx)
trajectory_history = np.array(trajectory_history)
theta_history = np.array(theta_history)
omega_history = np.array(omega_history)

# проверяем, на сколько "мягкая" стенка
contact_position = 5.0 - robot.radius

max_penetration = np.max(x) - contact_position
penetration_fraction = max_penetration / robot.radius

print(f"Maximum penetration: {max_penetration:.6f}")
print(f"Fraction of radius: {penetration_fraction:.2%}")

# продолжаем отрисовку движения бота
animation = animate_robots(
    robots=[robot],
    trajectory_history=trajectory_history,
    omega_history=omega_history,
    theta_history=theta_history,
    walls=[wall],
    box_size=(6, 2),
    interval=1,
    trail_length=500,
)

plt.show()

fig, axes = plt.subplots(2, 1, figsize=(8, 6))

axes[0].plot(time, x)
axes[0].axhline(4.5, color="red", linestyle="--",
                 label="Contact position")
axes[0].set_ylabel("x")
axes[0].legend()

axes[1].plot(time, vx)
axes[1].set_xlabel("Time")
axes[1].set_ylabel("vx")

plt.tight_layout()
plt.show()
