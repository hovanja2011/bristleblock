
import numpy as np
import matplotlib.pyplot as plt

from bristlebots.robot import Robot
from bristlebots.contacts import ContactParameters
from bristlebots.simulation import step_robots


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
    position=[5.0, 0.0],
    velocity=[-1.0, 0.0],
    radius=1.0,
    mass=1.0,
)

dt = 0.001
total_time = 3.0
steps = int(total_time / dt)

time = []
position_a = []
position_b = []
velocity_a = []
velocity_b = []
force_a = []
force_b = []

for i in range(steps):

    results = step_robots(
        [robot_a, robot_b],
        dt,
        contact_parameters=parameters,
    )

    time.append((i + 1) * dt)

    position_a.append(robot_a.position.copy())
    position_b.append(robot_b.position.copy())

    velocity_a.append(robot_a.velocity.copy())
    velocity_b.append(robot_b.velocity.copy())

    force_a.append(
        results[0]["robot_robot_contact_force"].copy()
    )

    force_b.append(
        results[1]["robot_robot_contact_force"].copy()
    )


time = np.array(time)
position_a = np.array(position_a)
position_b = np.array(position_b)
velocity_a = np.array(velocity_a)
velocity_b = np.array(velocity_b)
force_a = np.array(force_a)
force_b = np.array(force_b)


fig, axes = plt.subplots(3, 1, figsize=(10, 11))

# Positions
axes[0].plot(time, position_a[:, 0], label="Robot A")
axes[0].plot(time, position_b[:, 0], label="Robot B")
axes[0].set_ylabel("Position x")
axes[0].set_title("Robot positions")
axes[0].legend()
axes[0].grid()

# Velocities
axes[1].plot(time, velocity_a[:, 0], label="Robot A")
axes[1].plot(time, velocity_b[:, 0], label="Robot B")
axes[1].set_ylabel("Velocity vx")
axes[1].set_title("Robot velocities")
axes[1].legend()
axes[1].grid()

# Contact forces
axes[2].plot(time, force_a[:, 0], label="Force on A")
axes[2].plot(time, force_b[:, 0], label="Force on B")
axes[2].set_xlabel("Time")
axes[2].set_ylabel("Contact force Fx")
axes[2].set_title("Robot-robot contact forces")
axes[2].legend()
axes[2].grid()

plt.tight_layout()
plt.show()
