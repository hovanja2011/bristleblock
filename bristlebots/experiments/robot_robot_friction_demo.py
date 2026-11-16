
import numpy as np
import matplotlib.pyplot as plt

from bristlebots.robot import Robot
from bristlebots.contacts import ContactParameters
from bristlebots.simulation import step_robots
from bristlebots.visualization import animate_robots


# ==================================================
# 1. Parameters
# ==================================================

R = 1.0
DT = 1e-4
TOTAL_TIME = 0.12
STEPS = int(TOTAL_TIME / DT)

contact_parameters = ContactParameters(
    normal_stiffness=1000.0,
    normal_damping=2.0,
    mu_static=0.02,
    mu_kinetic=0.01,
    tangential_stiffness=100.0,
    tangential_damping=0.0,
)


# ==================================================
# 2. Initial configuration
# ==================================================

robot_a = Robot(
    position=(0.0, 0.0),
    velocity=(10.0, 0.0),
    theta=0.0,
    radius=R,
    mass=1.0,
    translational_drag=0.1,
    rotational_drag=0.1,
)

robot_b = Robot(
    position=(2.2, 0.0),
    velocity=(0.0, 0.0),
    theta=0.0,
    radius=R,
    mass=1.0,
    translational_drag=0.1,
    rotational_drag=0.1,
)

robots = [robot_a, robot_b]


# ==================================================
# 3. History
# ==================================================

trajectory_history = []
theta_history = []
omega_history = []

time_history = []

normal_force_history = []
tangential_force_history = []
overlap_history = []
mode_history = []
xi_history = []


# Initial state
trajectory_history.append(
    np.array([robot.position.copy() for robot in robots])
)

theta_history.append(
    np.array([robot.theta for robot in robots])
)

omega_history.append(
    np.array([robot.omega for robot in robots])
)


# ==================================================
# 4. Simulation
# ==================================================

for step in range(STEPS):

    diagnostics = step_robots(
        robots,
        dt=DT,
        contact_parameters=contact_parameters,
    )

    # Pairwise contact result for robot A and robot B
    _, result = diagnostics[0]["robot_contact_results"][0]

    # Geometry at the beginning of this integration step
    displacement = robot_b.position - robot_a.position
    distance = np.linalg.norm(displacement)

    if distance > 1e-12:
        normal = displacement / distance
    else:
        normal = np.array([1.0, 0.0])

    tangent = np.array([-normal[1], normal[0]])

    # Signed tangential force on robot A
    tangential_force = np.dot(
        result.force_on_first,
        tangent,
    )

    state = robot_a.robot_contact_states[robot_b]

    time_history.append((step + 1) * DT)

    normal_force_history.append(
        result.normal_force_magnitude
    )

    tangential_force_history.append(
        tangential_force
    )

    overlap_history.append(
        result.overlap
    )

    mode_history.append(
        result.mode
    )

    xi_history.append(
        state.tangential_displacement
    )

    # Save robot states after integration
    trajectory_history.append(
        np.array([
            robot.position.copy()
            for robot in robots
        ])
    )

    theta_history.append(
        np.array([
            robot.theta
            for robot in robots
        ])
    )

    omega_history.append(
        np.array([
            robot.omega
            for robot in robots
        ])
    )


# Convert histories to arrays
trajectory_history = np.array(trajectory_history)
theta_history = np.array(theta_history)
omega_history = np.array(omega_history)

time_history = np.array(time_history)

normal_force_history = np.array(normal_force_history)
tangential_force_history = np.array(tangential_force_history)
overlap_history = np.array(overlap_history)
xi_history = np.array(xi_history)
mode_history = np.array(mode_history)


# ==================================================
# 5. Existing animation
# ==================================================

animate_robots(
    robots=robots,
    trajectory_history=trajectory_history,
    omega_history=omega_history,
    theta_history=theta_history,
    box_size=(8, 6),
    interval=30,
    trail_length=100,
    show_trails=True,
    show_angular_velocity=True,
)


# ==================================================
# 6. Contact force plot
# ==================================================

mu_s = contact_parameters.mu_static
mu_k = contact_parameters.mu_kinetic

fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 9),
    sharex=True,
)

# Normal and tangential forces
axes[0].plot(
    time_history,
    normal_force_history,
    label="Normal force",
)

axes[0].plot(
    time_history,
    tangential_force_history,
    label="Tangential force",
)

axes[0].set_ylabel("Force")
axes[0].set_title("Robot-robot contact forces")
axes[0].legend()
axes[0].grid()


# Friction limits
axes[1].plot(
    time_history,
    tangential_force_history,
    label="Tangential force",
)

axes[1].plot(
    time_history,
    mu_s * normal_force_history,
    "--",
    label="Static limit (+)",
)

axes[1].plot(
    time_history,
    -mu_s * normal_force_history,
    "--",
    label="Static limit (-)",
)

axes[1].plot(
    time_history,
    mu_k * normal_force_history,
    ":",
    label="Kinetic limit (+)",
)

axes[1].plot(
    time_history,
    -mu_k * normal_force_history,
    ":",
    label="Kinetic limit (-)",
)

axes[1].set_ylabel("Force")
axes[1].set_title("Coulomb friction limits")
axes[1].legend()
axes[1].grid()


# Contact mode
mode_numbers = {
    "no_contact": 0,
    "no_load": 1,
    "sticking": 2,
    "sliding": 3,
}

mode_values = np.array([
    mode_numbers[str(mode)]
    for mode in mode_history
])

axes[2].step(
    time_history,
    mode_values,
    where="post",
)

axes[2].set_yticks([0, 1, 2, 3])
axes[2].set_yticklabels([
    "No contact",
    "No load",
    "Sticking",
    "Sliding",
])

axes[2].set_xlabel("Time")
axes[2].set_title("Contact mode")
axes[2].grid()

plt.tight_layout()
plt.show()


# ==================================================
# 7. Tangential displacement and overlap
# ==================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(10, 6),
    sharex=True,
)

axes[0].plot(
    time_history,
    xi_history,
)

axes[0].set_ylabel("Tangential displacement")
axes[0].set_title("Tangential contact history")
axes[0].grid()

axes[1].plot(
    time_history,
    overlap_history,
)

axes[1].set_xlabel("Time")
axes[1].set_ylabel("Overlap")
axes[1].set_title("Contact overlap")
axes[1].grid()

plt.tight_layout()
plt.show()
