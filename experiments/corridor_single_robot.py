
import numpy as np
import matplotlib.pyplot as plt

from bristlebots.robot import Robot
from bristlebots.geometry import Wall
from bristlebots.contacts import ContactParameters
from bristlebots.simulation import step


# ==================================================
# Geometry (all lengths are measured in robot radii)
# ==================================================

RADIUS = 1.0

CORRIDOR_WIDTH = 5.0 * RADIUS
CORRIDOR_LENGTH = 40.0 * RADIUS

INITIAL_X = 5.0 * RADIUS
INITIAL_Y = CORRIDOR_WIDTH / 2

# ==================================================
# Robot parameters
# ==================================================

MASS = 1.0
V0 = 1.0

INITIAL_ANGLE = np.deg2rad(20.0)
INITIAL_SPEED = V0

# Constant propulsion and linear drag:
# free swimming speed = propulsion_force / translational_drag

TRANSLATIONAL_DRAG = 1.0
PROPULSION_FORCE = TRANSLATIONAL_DRAG * V0

ROTATIONAL_DRAG = 0.1

# ==================================================
# Contact parameters
# ==================================================

NORMAL_STIFFNESS = 1.0e4
NORMAL_DAMPING = 20.0

TANGENTIAL_STIFFNESS = 1.0e4

MU_STATIC = 0.5
MU_KINETIC = 0.3

# ==================================================
# Numerical parameters
# ==================================================

DT = 5.0e-4
TOTAL_TIME = 25.0

# ==================================================
# Geometry and robot initialization
# ==================================================


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


def create_robot():
    velocity = INITIAL_SPEED * np.array([
        np.cos(INITIAL_ANGLE),
        np.sin(INITIAL_ANGLE),
    ])

    return Robot(
        position=np.array([INITIAL_X, INITIAL_Y]),
        velocity=velocity,
        theta=INITIAL_ANGLE,
        omega=0.0,
        radius=RADIUS,
        mass=MASS,
        propulsion_force=PROPULSION_FORCE,
        translational_drag=TRANSLATIONAL_DRAG,
        rotational_drag=ROTATIONAL_DRAG,
    )


def create_contact_parameters(friction_enabled):
    if friction_enabled:
        mu_static = MU_STATIC
        mu_kinetic = MU_KINETIC
    else:
        mu_static = 0.0
        mu_kinetic = 0.0

    return ContactParameters(
        normal_stiffness=NORMAL_STIFFNESS,
        normal_damping=NORMAL_DAMPING,
        mu_static=mu_static,
        mu_kinetic=mu_kinetic,
        tangential_stiffness=TANGENTIAL_STIFFNESS,
    )


# ==================================================
# Simulation
# ==================================================


def run_experiment(friction_enabled):
    robot = create_robot()
    walls = create_corridor()

    contact_parameters = create_contact_parameters(
        friction_enabled
    )

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


# ==================================================
# Run both experiments
# ==================================================


if __name__ == "__main__":

    print("Running frictionless experiment...")
    no_friction = run_experiment(friction_enabled=False)

    print("Running experiment with Coulomb friction...")
    with_friction = run_experiment(friction_enabled=True)

    # --------------------------------------------------
    # Plot trajectories
    # --------------------------------------------------

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for ax, history, title in [
        (axes[0, 0], no_friction, "Without friction"),
        (axes[0, 1], with_friction, "With Coulomb friction"),
    ]:
        ax.plot(
            history["position"][:, 0],
            history["position"][:, 1],
        )

        ax.axhline(0.0, color="black")
        ax.axhline(CORRIDOR_WIDTH, color="black")

        ax.set_xlim(0, CORRIDOR_LENGTH)
        ax.set_ylim(-0.5, CORRIDOR_WIDTH + 0.5)

        ax.set_aspect("equal")
        ax.set_xlabel("x / R")
        ax.set_ylabel("y / R")
        ax.set_title(title)
        ax.grid(True)

    # --------------------------------------------------
    # Speed
    # --------------------------------------------------

    axes[1, 0].plot(
        no_friction["time"],
        no_friction["speed"],
        label="Without friction",
    )

    axes[1, 0].plot(
        with_friction["time"],
        with_friction["speed"],
        label="With friction",
    )

    axes[1, 0].axhline(V0, color="black", linestyle="--",
                       label="Free speed")

    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("|v| / v0")
    axes[1, 0].set_title("Robot speed")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # --------------------------------------------------
    # Angular velocity
    # --------------------------------------------------

    axes[1, 1].plot(
        no_friction["time"],
        no_friction["omega"],
        label="Without friction",
    )

    axes[1, 1].plot(
        with_friction["time"],
        with_friction["omega"],
        label="With friction",
    )

    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel("Angular velocity")
    axes[1, 1].set_title("Angular velocity")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # Orientation
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.plot(
        no_friction["time"],
        np.rad2deg(no_friction["theta"]),
        label="Without friction",
    )

    ax.plot(
        with_friction["time"],
        np.rad2deg(with_friction["theta"]),
        label="With friction",
    )

    ax.axhline(90.0, color="black", linestyle="--")

    ax.set_xlabel("Time")
    ax.set_ylabel("Orientation (degrees)")
    ax.set_title("Robot orientation")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()