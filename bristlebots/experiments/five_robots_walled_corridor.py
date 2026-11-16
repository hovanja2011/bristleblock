
import numpy as np
import matplotlib.pyplot as plt

from bristlebots.robot import Robot
from bristlebots.geometry import Wall
from bristlebots.contacts import ContactParameters
from bristlebots.simulation import step_robots
from bristlebots.visualization import animate_robots


# ==================================================
# 1. Parameters
# ==================================================

R = 1.0

CORRIDOR_LENGTH = 30 * R
CORRIDOR_WIDTH = 5 * R

DT = 1e-4
TOTAL_TIME = 12.0
STEPS = int(TOTAL_TIME / DT)

SAVE_EVERY = 30

# External force acting on every robot
EXTERNAL_FORCE = np.array([0.03, 0.0])

# Random initial angular velocities
RNG = np.random.default_rng(seed=42)

OMEGA_MIN = -0.5
OMEGA_MAX = 0.5

initial_omegas = RNG.uniform(
    OMEGA_MIN,
    OMEGA_MAX,
    size=5,
)

ANGLE_MIN = -np.pi / 2
ANGLE_MAX = np.pi / 2

initial_angles = RNG.uniform(
    ANGLE_MIN,
    ANGLE_MAX,
    size=5,
)

# Ensure all initial angular velocities are different
while len(np.unique(initial_omegas)) != 5:
    initial_omegas = RNG.uniform(
        OMEGA_MIN,
        OMEGA_MAX,
        size=5,
    )


contact_parameters = ContactParameters(
    normal_stiffness=2000.0,
    normal_damping=4.0,
    mu_static=0.4,
    mu_kinetic=0.3,
    tangential_stiffness=500.0,
    tangential_damping=0.5,
)


# ==================================================
# 2. Walls
# ==================================================

walls = [
    Wall(
        start=np.array([0.0, 0.0]),
        end=np.array([CORRIDOR_LENGTH, 0.0]),
    ),
    Wall(
        start=np.array([0.0, CORRIDOR_WIDTH]),
        end=np.array([CORRIDOR_LENGTH, CORRIDOR_WIDTH]),
    ),
]


# ==================================================
# 3. Initial conditions
# ==================================================

initial_positions = [
    (2.5, 1.5),
    (4.5, 3.5),
    (6.5, 1.5),
    (8.5, 3.5),
    (10.5, 1.5),
]

initial_speeds = [
    1.00,
    0.65,
    1.05,
    0.70,
    1.00,
]

robots = []

for position, speed, omega, angle in zip(
    initial_positions,
    initial_speeds,
    initial_omegas,
    initial_angles
):

    robot = Robot(
        position=position,
        velocity=speed * np.array([
            np.cos(angle),
            np.sin(angle),
        ]),
        theta=angle,
        omega=omega,
        radius=R,
        mass=1.0,
        propulsion_force=0.15,
        translational_drag=0.1,
        rotational_drag=0.05,
    )

    robots.append(robot)


print("Initial angular velocities:")

for i, robot in enumerate(robots):
    print(f"Robot {i + 1}: omega = {robot.omega:.4f}")


# ==================================================
# 4. Histories
# ==================================================

trajectory_history = []
theta_history = []
omega_history = []

time_history = []

pair_contact_history = []
wall_contact_history = []
max_overlap_history = []


def save_robot_state():

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


save_robot_state()


# ==================================================
# 5. Simulation
# ==================================================

for step in range(STEPS):

    diagnostics = step_robots(
        robots,
        dt=DT,
        walls=walls,
        contact_parameters=contact_parameters,
        external_force=EXTERNAL_FORCE,
    )

    pair_contacts = 0
    wall_contacts = 0
    overlaps = []

    for i, diag in enumerate(diagnostics):

        # Robot-wall contacts
        for result in diag["contact_results"]:

            overlaps.append(result.overlap)

            if result.overlap > 0:
                wall_contacts += 1

        # Robot-robot contacts
        for other_robot, result in diag["robot_contact_results"]:

            j = robots.index(other_robot)

            # Count each pair only once
            if j > i:

                overlaps.append(result.overlap)

                if result.overlap > 0:
                    pair_contacts += 1

    max_overlap = max(overlaps, default=0.0)

    time = (step + 1) * DT

    time_history.append(time)
    pair_contact_history.append(pair_contacts)
    wall_contact_history.append(wall_contacts)
    max_overlap_history.append(max_overlap)

    if (step + 1) % SAVE_EVERY == 0 or step == STEPS - 1:
        save_robot_state()


# ==================================================
# 6. Convert histories
# ==================================================

trajectory_history = np.asarray(trajectory_history)
theta_history = np.asarray(theta_history)
omega_history = np.asarray(omega_history)

time_history = np.asarray(time_history)

pair_contact_history = np.asarray(pair_contact_history)
wall_contact_history = np.asarray(wall_contact_history)
max_overlap_history = np.asarray(max_overlap_history)


# ==================================================
# 7. Numerical summary
# ==================================================

print("\nSimulation finished")

print(f"Number of robots: {len(robots)}")

print(f"External force: {EXTERNAL_FORCE}")

print(
    f"Maximum robot-robot contacts: "
    f"{np.max(pair_contact_history)}"
)

print(
    f"Maximum robot-wall contacts: "
    f"{np.max(wall_contact_history)}"
)

print(
    f"Maximum overlap: "
    f"{np.max(max_overlap_history):.6f}"
)

print(
    f"Maximum relative overlap: "
    f"{np.max(max_overlap_history) / R:.4%}"
)


# ==================================================
# 8. Contact diagnostics
# ==================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(10, 7),
    sharex=True,
)

axes[0].plot(
    time_history,
    pair_contact_history,
)

axes[0].set_ylabel("Number of contacts")
axes[0].set_title("Robot-robot contacts")
axes[0].grid()

axes[1].plot(
    time_history,
    wall_contact_history,
)

axes[1].set_xlabel("Time")
axes[1].set_ylabel("Number of contacts")
axes[1].set_title("Robot-wall contacts")
axes[1].grid()

plt.tight_layout()
plt.show()


# ==================================================
# 9. Overlap
# ==================================================

plt.figure(figsize=(10, 4))

plt.plot(
    time_history,
    max_overlap_history / R,
)

plt.xlabel("Time")
plt.ylabel("Maximum overlap / R")
plt.title("Maximum relative overlap")
plt.grid()

plt.tight_layout()
plt.show()


# ==================================================
# 10. Animation
# ==================================================

animation = animate_robots(
    robots=robots,
    trajectory_history=trajectory_history,
    omega_history=omega_history,
    theta_history=theta_history,
    walls=walls,
    box_size=(CORRIDOR_LENGTH, CORRIDOR_WIDTH),
    interval=10,
    trail_length=150,
    show_trails=True,
    show_angular_velocity=True,
    show_propulsion_point=True,
)

plt.show()
