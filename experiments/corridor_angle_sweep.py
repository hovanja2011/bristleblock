
import numpy as np
import matplotlib.pyplot as plt

from bristlebots.robot import Robot
from bristlebots.geometry import Wall
from bristlebots.contacts import ContactParameters
from bristlebots.simulation import step


# ==================================================
# Geometry: lengths are measured in robot radii
# ==================================================

RADIUS = 1.0

CORRIDOR_WIDTH = 5.0 * RADIUS
CORRIDOR_LENGTH = 60.0 * RADIUS

INITIAL_X = 5.0 * RADIUS
INITIAL_Y = CORRIDOR_WIDTH / 2

# ==================================================
# Robot parameters
# ==================================================

MASS = 1.0
V0 = 1.0

TRANSLATIONAL_DRAG = 1.0
PROPULSION_FORCE = TRANSLATIONAL_DRAG * V0

ROTATIONAL_DRAG = 0.1

# ==================================================
# Contact parameters
# ==================================================

NORMAL_STIFFNESS = 1.0e4
NORMAL_DAMPING = 20.0

TANGENTIAL_STIFFNESS = 1.0e4
TANGENTIAL_DAMPING = 0.0

MU_STATIC = 0.5
MU_KINETIC = 0.3

# ==================================================
# Numerical parameters
# ==================================================

DT = 5.0e-4
TOTAL_TIME = 40.0

INITIAL_ANGLES_DEG = [5, 20, 35, 50, 70, 85]


def create_corridor():
    lower_wall = Wall(
        start=np.array([0.0, 0.0]),
        end=np.array([CORRIDOR_LENGTH, 0.0]),
    )

    upper_wall = Wall(
        start=np.array([0.0, CORRIDOR_WIDTH]),
        end=np.array([CORRIDOR_LENGTH, CORRIDOR_WIDTH]),
    )

    return [lower_wall, upper_wall]


def create_robot(initial_angle):
    theta = np.deg2rad(initial_angle)

    velocity = V0 * np.array([
        np.cos(theta),
        np.sin(theta),
    ])

    return Robot(
        position=np.array([INITIAL_X, INITIAL_Y]),
        velocity=velocity,
        theta=theta,
        omega=0.0,
        radius=RADIUS,
        mass=MASS,
        propulsion_force=PROPULSION_FORCE,
        translational_drag=TRANSLATIONAL_DRAG,
        rotational_drag=ROTATIONAL_DRAG,
    )


def create_contact_parameters():
    return ContactParameters(
        normal_stiffness=NORMAL_STIFFNESS,
        normal_damping=NORMAL_DAMPING,
        mu_static=MU_STATIC,
        mu_kinetic=MU_KINETIC,
        tangential_stiffness=TANGENTIAL_STIFFNESS,
        tangential_damping=TANGENTIAL_DAMPING,
    )


def run_experiment(initial_angle):
    robot = create_robot(initial_angle)
    walls = create_corridor()
    contact_parameters = create_contact_parameters()

    steps = int(round(TOTAL_TIME / DT))

    history = {
        "time": [],
        "position": [],
        "velocity": [],
        "speed": [],
        "theta": [],
        "omega": [],
    }

    for i in range(steps):
        history["time"].append(i * DT)
        history["position"].append(robot.position.copy())
        history["velocity"].append(robot.velocity.copy())
        history["speed"].append(np.linalg.norm(robot.velocity))
        history["theta"].append(robot.theta)
        history["omega"].append(robot.omega)

        step(
            robot,
            DT,
            walls=walls,
            contact_parameters=contact_parameters,
        )

    for key in history:
        history[key] = np.asarray(history[key])

    return history


if __name__ == "__main__":

    results = {}

    for angle in INITIAL_ANGLES_DEG:
        print(f"Running initial angle {angle} degrees...")
        results[angle] = run_experiment(angle)

    # ==================================================
    # Print final-state summary
    # ==================================================

    print("\nFinal states")
    print("-" * 75)
    print(
        f"{'Initial angle':>14} "
        f"{'Final angle':>14} "
        f"{'Final speed':>14} "
        f"{'Final omega':>14} "
        f"{'Final y':>12}"
    )
    print("-" * 75)

    for angle, history in results.items():
        final_angle = np.rad2deg(history["theta"][-1])
        final_speed = history["speed"][-1]
        final_omega = history["omega"][-1]
        final_y = history["position"][-1, 1]

        print(
            f"{angle:14.1f} "
            f"{final_angle:14.2f} "
            f"{final_speed:14.4f} "
            f"{final_omega:14.5f} "
            f"{final_y:12.4f}"
        )

    # ==================================================
    # Plot trajectories
    # ==================================================

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]

    for angle, history in results.items():
        position = history["position"]

        ax.plot(
            position[:, 0],
            position[:, 1],
            label=f"{angle} deg",
        )

    ax.axhline(0.0, color="black")
    ax.axhline(CORRIDOR_WIDTH, color="black")

    ax.set_xlim(0, CORRIDOR_LENGTH)
    ax.set_ylim(-0.5, CORRIDOR_WIDTH + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x / R")
    ax.set_ylabel("y / R")
    ax.set_title("Robot trajectories")
    ax.legend()
    ax.grid(True)

    # ==================================================
    # Orientation over time
    # ==================================================

    ax = axes[0, 1]

    for angle, history in results.items():
        ax.plot(
            history["time"],
            np.rad2deg(history["theta"]),
            label=f"{angle} deg",
        )

    ax.axhline(90.0, color="black", linestyle="--")

    ax.set_xlabel("Time")
    ax.set_ylabel("Orientation (degrees)")
    ax.set_title("Robot orientation")
    ax.legend()
    ax.grid(True)

    # ==================================================
    # Final orientation vs initial orientation
    # ==================================================

    ax = axes[1, 0]

    final_angles = [
        np.rad2deg(history["theta"][-1])
        for history in results.values()
    ]

    ax.plot(
        INITIAL_ANGLES_DEG,
        final_angles,
        "o-",
    )

    ax.plot(
        [0, 90],
        [0, 90],
        "--",
        color="gray",
        label="No reorientation",
    )

    ax.axhline(90.0, color="black", linestyle=":")

    ax.set_xlabel("Initial angle (degrees)")
    ax.set_ylabel("Final angle (degrees)")
    ax.set_title("Final orientation")
    ax.legend()
    ax.grid(True)

    # ==================================================
    # Speed over time
    # ==================================================

    ax = axes[1, 1]

    for angle, history in results.items():
        ax.plot(
            history["time"],
            history["speed"],
            label=f"{angle} deg",
        )

    ax.axhline(V0, color="black", linestyle="--")

    ax.set_xlabel("Time")
    ax.set_ylabel("Speed / v0")
    ax.set_title("Robot speed")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()
