
import numpy as np
import matplotlib.pyplot as plt

from bristlebots.contacts import ContactParameters
from bristlebots.geometry import Wall
from bristlebots.robot import Robot
from bristlebots.simulation import step
from bristlebots.visualization import animate_robots

# 1. Geometry
wall_position = 2.0
wall = Wall(
    start=[wall_position, -1.0],
    end=[wall_position, 5.0],
)

# 2. Robot
robot = Robot(
    position=[0.0, 0.0],
    velocity=[20.0, 5.0],
    theta=0.0,
    omega=15.0,
    radius=0.5,
    mass=1.0,
    propulsion_force=0.0,
    translational_drag=0.0,
    rotational_drag=0.0,
)

# 3. Contact parameters
k_n = 4e5
eta_n = 40.0

mu_static = 0.3
mu_kinetic = 0.3

k_t = 1e4

contact_parameters = ContactParameters(
    normal_stiffness=k_n,
    normal_damping=eta_n,
    mu_static=mu_static,
    mu_kinetic=mu_kinetic,
    tangential_stiffness=k_t,
)
# 4. Simulation parameters

dt = 5e-5
total_time = 2.0

steps = int(total_time / dt)

# 5. Data storage
time = [0.0]

x = [robot.position[0]]
vx = [robot.velocity[0]]

trajectory_history = [
    np.array([robot.position.copy()])
]

theta_history = [[robot.theta]]
omega_history = [[robot.omega]]

for i in range(steps):

    # Advance the physical simulation by one time step
    step(
        robot,
        dt,
        walls=[wall],
        contact_parameters=contact_parameters,
    )

    # Save simulation data
    time.append((i + 1) * dt)

    x.append(robot.position[0])
    vx.append(robot.velocity[0])

    trajectory_history.append(
        np.array([robot.position.copy()])
    )

    theta_history.append([robot.theta])
    omega_history.append([robot.omega])

# 7. Convert data to NumPy arrays
time = np.array(time)
x = np.array(x)
vx = np.array(vx)

trajectory_history = np.array(trajectory_history)
theta_history = np.array(theta_history)
omega_history = np.array(omega_history)

# 8. Analyze collision
contact_position = wall_position - robot.radius

max_penetration = np.max(x) - contact_position
penetration_fraction = max_penetration / robot.radius

print(f"Maximum penetration: {max_penetration:.6f}")
print(f"Fraction of radius: {penetration_fraction:.2%}")

# 9. Animation
animation_stride = 20

animation = animate_robots(
    robots=[robot],
    trajectory_history=trajectory_history[::animation_stride],
    omega_history=omega_history[::animation_stride],
    theta_history=theta_history[::animation_stride],
    walls=[wall],
    box_size=(wall_position + 1, 6),
    interval=20,
    trail_length=500,
)

# 10. Plot results
fig, axes = plt.subplots(2, 1, figsize=(8, 6))

axes[0].plot(time, x)

axes[0].axhline(
    contact_position,
    color="red",
    linestyle="--",
    label="Contact position",
)

axes[0].set_ylabel("x")
axes[0].legend()

axes[1].plot(time, vx)

axes[1].set_xlabel("Time")
axes[1].set_ylabel("vx")

plt.tight_layout()

plt.show()
