
import numpy as np
import matplotlib.pyplot as plt

from bristlebots.robot import Robot
from bristlebots.contacts import ContactParameters
from bristlebots.simulation import step_robots
from bristlebots.visualization import animate_robots


def run_simulation(
    robots,
    dt,
    steps,
    contact_parameters,
    save_every=10,
):
    """
    Run robot simulation and collect diagnostic histories.

    Returns a dictionary containing simulation histories.
    """

    # ----------------------------------------------
    # Initialize histories
    # ----------------------------------------------

    history = {
        "trajectory": [],
        "theta": [],
        "omega": [],
        "time": [],
        "normal_force": [],
        "tangential_force": [],
        "overlap": [],
        "mode": [],
        "xi": [],
        "active_torque": [],
        "contact_torque_a": [],
        "contact_torque_b": [],
        "kinetic_energy": [],
        "angular_velocity_a": [],
        "angular_velocity_b": [],
        "state_time": [],
    }

    robot_a, robot_b = robots

    # ----------------------------------------------
    # Save robot state
    # ----------------------------------------------

    def save_robot_state(time):
        history["state_time"].append(time)
        history["trajectory"].append(
            np.array([
                robot.position.copy()
                for robot in robots
            ])
        )

        history["theta"].append(
            np.array([
                robot.theta
                for robot in robots
            ])
        )

        history["omega"].append(
            np.array([
                robot.omega
                for robot in robots
            ])
        )

    save_robot_state(0.0)

    # ----------------------------------------------
    # Main simulation loop
    # ----------------------------------------------

    for step in range(steps):

        # Geometry before integration
        displacement = robot_b.position - robot_a.position
        distance = np.linalg.norm(displacement)

        if distance > 1e-12:
            normal = displacement / distance
        else:
            normal = np.array([1.0, 0.0])

        tangent = np.array([-normal[1], normal[0]])

        # Integrate all robots
        diagnostics = step_robots(
            robots,
            dt=dt,
            contact_parameters=contact_parameters,
        )

        # Pairwise contact result
        _, result = diagnostics[0]["robot_contact_results"][0]

        # Signed tangential force on robot A
        ft = np.dot(result.force_on_first, tangent)

        state = robot_a.robot_contact_states[robot_b]

        # Time
        time = (step + 1) * dt
        history["time"].append(time)

        # Contact diagnostics
        history["normal_force"].append(
            result.normal_force_magnitude
        )

        history["tangential_force"].append(ft)
        history["overlap"].append(result.overlap)
        history["mode"].append(result.mode)

        history["xi"].append(
            state.tangential_displacement
        )

        # Active torques
        history["active_torque"].append([
            diagnostics[0]["active_torque"],
            diagnostics[1]["active_torque"],
        ])

        # Contact torques
        history["contact_torque_a"].append(
            result.torque_on_first
        )

        history["contact_torque_b"].append(
            result.torque_on_second
        )

        # Total translational + rotational kinetic energy
        kinetic_energy = 0.0

        for robot in robots:

            kinetic_energy += (
                0.5 * robot.mass
                * np.dot(robot.velocity, robot.velocity)
            )

            kinetic_energy += (
                0.5 * robot.moment_of_inertia
                * robot.omega**2
            )

        history["kinetic_energy"].append(kinetic_energy)

        history["angular_velocity_a"].append(robot_a.omega)
        history["angular_velocity_b"].append(robot_b.omega)

        # Save visualization state
        if (step + 1) % save_every == 0:
            save_robot_state(time)

    # ----------------------------------------------
    # Convert histories to NumPy arrays
    # ----------------------------------------------

    for key in history:
        history[key] = np.asarray(history[key])

    return history

# ==================================================
# 1. Parameters
# ==================================================

R = 1.0
DT = 1e-4
TOTAL_TIME = 4.0
STEPS = int(TOTAL_TIME / DT)

contact_parameters = ContactParameters(
    normal_stiffness=2000.0,
    normal_damping=4.0,
    mu_static=0.4,
    mu_kinetic=0.3,
    tangential_stiffness=500.0,
    tangential_damping=0.5,
)


# ==================================================
# 2. Initial conditions
# ==================================================

robot_a = Robot(
    position=(0.0, 0.0),
    velocity=(1.0, 0.05),
    theta=0.0,
    omega=0.5,
    radius=R,
    mass=1.0,
    propulsion_force=0.15,
    active_torque=0.1275,
    translational_drag=0.1,
    rotational_drag=0.05,
)

robot_b = Robot(
    position=(2.0, 1.2),
    velocity=(-0.1, 0.0),
    theta=np.pi,
    omega=-0.3,
    radius=R,
    mass=1.0,
    propulsion_force=0.15,
    active_torque=-0.0225,
    translational_drag=0.1,
    rotational_drag=0.05,
)

# ==================================================
# 3. Run simulation
# ==================================================

robots = [robot_a, robot_b]

history = run_simulation(
    robots=robots,
    dt=DT,
    steps=STEPS,
    contact_parameters=contact_parameters,
    save_every=10,
)


# ==================================================
# 4. Extract histories
# ==================================================

trajectory_history = history["trajectory"]
theta_history = history["theta"]
omega_history = history["omega"]

time_history = history["time"]

normal_force_history = history["normal_force"]
tangential_force_history = history["tangential_force"]
overlap_history = history["overlap"]
mode_history = history["mode"]
xi_history = history["xi"]

active_torque_history = history["active_torque"]
contact_torque_a_history = history["contact_torque_a"]
contact_torque_b_history = history["contact_torque_b"]

kinetic_energy_history = history["kinetic_energy"]

# ==================================================
# 6. Numerical summary
# ==================================================

max_overlap = np.max(overlap_history)

print("Simulation finished")
print(f"Maximum overlap: {max_overlap:.6f}")
print(f"Maximum relative overlap: {max_overlap / R:.4%}")

print("\nContact modes:")

for mode in [
    "no_contact",
    "no_load",
    "sticking",
    "sliding",
]:
    count = np.count_nonzero(mode_history == mode)
    print(f"{mode}: {count} steps")

print("\nFinal angular velocities:")
print(f"Robot A: {robot_a.omega:.6f}")
print(f"Robot B: {robot_b.omega:.6f}")


# ==================================================
# 7. Contact forces
# ==================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(11, 9),
    sharex=True,
)

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
axes[0].set_title("Contact forces")
axes[0].legend()
axes[0].grid()


# Coulomb limits
mu_s = contact_parameters.mu_static
mu_k = contact_parameters.mu_kinetic

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
axes[1].set_title("Coulomb friction")
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
# 8. Rotational dynamics
# ==================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(10, 7),
    sharex=True,
)

axes[0].plot(
    history["state_time"],
    omega_history[:, 0],
    label="Robot A",
)

axes[0].plot(
    history["state_time"],
    omega_history[:, 1],
    label="Robot B",
)

axes[0].set_ylabel("Angular velocity")
axes[0].set_title("Angular velocities")
axes[0].legend()
axes[0].grid()


# Compare active torque and contact torques
axes[1].plot(
    time_history,
    active_torque_history[:, 0],
    label="Active torque A",
)

axes[1].plot(
    time_history,
    active_torque_history[:, 1],
    label="Active torque B",
)

axes[1].plot(
    time_history,
    contact_torque_a_history,
    label="Contact torque A",
)

axes[1].plot(
    time_history,
    contact_torque_b_history,
    label="Contact torque B",
)

axes[1].set_xlabel("Time")
axes[1].set_ylabel("Torque")
axes[1].set_title("Origin of rotational motion")
axes[1].legend()
axes[1].grid()

plt.tight_layout()
plt.show()


# ==================================================
# 9. Overlap, tangential history and energy
# ==================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 9),
    sharex=True,
)

axes[0].plot(
    time_history,
    overlap_history / R,
)

axes[0].set_ylabel("Overlap / R")
axes[0].set_title("Relative overlap")
axes[0].grid()


axes[1].plot(
    time_history,
    xi_history,
)

axes[1].set_ylabel("Tangential displacement")
axes[1].set_title("Contact history")
axes[1].grid()


axes[2].plot(
    time_history,
    kinetic_energy_history,
)

axes[2].set_xlabel("Time")
axes[2].set_ylabel("Kinetic energy")
axes[2].set_title("Total kinetic energy")
axes[2].grid()

plt.tight_layout()
plt.show()


# ==================================================
# 10. Animation using existing visualization
# ==================================================

animation = animate_robots(
    robots=robots,
    trajectory_history=trajectory_history,
    omega_history=omega_history,
    theta_history=theta_history,
    box_size=(8, 6),
    interval=30,
    trail_length=150,
    show_trails=True,
    show_angular_velocity=True,
)

plt.show()
