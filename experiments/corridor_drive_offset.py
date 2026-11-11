
import numpy as np
import matplotlib.pyplot as plt

from bristlebots.robot import Robot
from bristlebots.geometry import Wall
from bristlebots.contacts import ContactParameters
from bristlebots.simulation import step
from bristlebots.visualization import animate_robots


# ============================================================
# 1. Geometry
# ============================================================

RADIUS = 1.0
MASS = 1.0

CORRIDOR_WIDTH = 5.0 * RADIUS
CORRIDOR_LENGTH = 50.0 * RADIUS

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


# ============================================================
# 2. Robot parameters
# ============================================================

INITIAL_POSITION = np.array([5.0, 2.5])

INITIAL_ANGLE = np.deg2rad(20.0)
INITIAL_SPEED = 1.0

INITIAL_VELOCITY = INITIAL_SPEED * np.array([
    np.cos(INITIAL_ANGLE),
    np.sin(INITIAL_ANGLE),
])

PROPULSION_FORCE = 1.0
TRANSLATIONAL_DRAG = 1.0
ROTATIONAL_DRAG = 0.1


# ============================================================
# 3. Contact parameters
# ============================================================

MU_K = 0.005
MU_S = (5.0 / 3.0) * MU_K

contact_parameters = ContactParameters(
    normal_stiffness=1e4,
    normal_damping=20.0,
    mu_static=MU_S,
    mu_kinetic=MU_K,
    tangential_stiffness=1e4,
    tangential_damping=0.0,
)


# ============================================================
# 4. Numerical parameters
# ============================================================

DT = 5e-4
TOTAL_TIME = 40.0

TIME = np.arange(0.0, TOTAL_TIME + DT, DT)

# Signed offsets, measured in units of R.
OFFSET_VALUES = [
    -0.02,
    -0.01,
    0.0,
    0.01,
    0.02,
]


# ============================================================
# 5. Simulation
# ============================================================

def run_simulation(offset):

    robot = Robot(
        position=INITIAL_POSITION.copy(),
        velocity=INITIAL_VELOCITY.copy(),
        theta=INITIAL_ANGLE,
        omega=0.0,
        radius=RADIUS,
        mass=MASS,
        propulsion_force=PROPULSION_FORCE,
        translational_drag=TRANSLATIONAL_DRAG,
        rotational_drag=ROTATIONAL_DRAG,
        propulsion_offset=np.array([0.0, offset * RADIUS]),
    )

    position_history = np.zeros((len(TIME), 2))
    velocity_history = np.zeros((len(TIME), 2))
    theta_history = np.zeros(len(TIME))
    omega_history = np.zeros(len(TIME))
    speed_history = np.zeros(len(TIME))

    for i in range(len(TIME)):

        position_history[i] = robot.position
        velocity_history[i] = robot.velocity
        theta_history[i] = robot.theta
        omega_history[i] = robot.omega
        speed_history[i] = np.linalg.norm(robot.velocity)

        if i == len(TIME) - 1:
            break

        step(
            robot,
            DT,
            walls=walls,
            contact_parameters=contact_parameters,
        )

    return {
        "offset": offset,
        "robot": robot,
        "position": position_history,
        "velocity": velocity_history,
        "theta": theta_history,
        "omega": omega_history,
        "speed": speed_history,
    }


# ============================================================
# 6. Run sweep
# ============================================================

results = []

for offset in OFFSET_VALUES:

    print(f"Running b/R = {offset:.3f}")

    results.append(run_simulation(offset))


# ============================================================
# 7. Final-state table
# ============================================================

print("\nFinal-state averages")
print("-" * 75)
print(
    f"{'b / R':>10} {'Angle':>12} "
    f"{'Speed':>12} {'Omega':>12} {'y / R':>12}"
)
print("-" * 75)

final_start = int(0.9 * len(TIME))

for r in results:

    theta_deg = np.rad2deg(r["theta"])

    final_angle = np.mean(theta_deg[final_start:])
    final_speed = np.mean(r["speed"][final_start:])
    final_omega = np.mean(r["omega"][final_start:])
    final_y = np.mean(r["position"][final_start:, 1])

    print(
        f"{r['offset']:10.3f} "
        f"{final_angle:12.3f} "
        f"{final_speed:12.5f} "
        f"{final_omega:12.5f} "
        f"{final_y / RADIUS:12.5f}"
    )


# ============================================================
# 8. Plot trajectories
# ============================================================

fig, ax = plt.subplots(figsize=(11, 5))

for r in results:

    ax.plot(
        r["position"][:, 0] / RADIUS,
        r["position"][:, 1] / RADIUS,
        label=rf"$b/R={r['offset']}$",
    )

ax.axhline(0.0, color="black")
ax.axhline(CORRIDOR_WIDTH / RADIUS, color="black")

ax.set_xlabel(r"$x/R$")
ax.set_ylabel(r"$y/R$")
ax.set_title("Robot trajectories in the corridor")
ax.grid(True)
ax.legend()

fig.tight_layout()


# ============================================================
# 9. Plot orientation, speed and angular velocity
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(10, 11))

for r in results:

    label = rf"$b/R={r['offset']}$"

    axes[0].plot(
        TIME,
        np.rad2deg(r["theta"]),
        label=label,
    )

    axes[1].plot(
        TIME,
        r["speed"],
        label=label,
    )

    axes[2].plot(
        TIME,
        r["omega"],
        label=label,
    )

axes[0].set_ylabel("Orientation (deg)")
axes[1].set_ylabel("Speed")
axes[2].set_ylabel("Angular velocity")
axes[2].set_xlabel("Time")

for ax in axes:
    ax.grid(True)
    ax.legend()

fig.suptitle("Effect of propulsion offset")
fig.tight_layout()


# ============================================================
# 10. Animation using the visualization module
# ============================================================

# Robot objects (one from each independent simulation).
robots = [
    r["robot"]
    for r in results
]

# Combine histories:
# positions: (n_frames, n_robots, 2)
# angles:    (n_frames, n_robots)
# omega:     (n_frames, n_robots)

trajectory_history = np.stack(
    [r["position"] for r in results],
    axis=1,
)

theta_history = np.stack(
    [r["theta"] for r in results],
    axis=1,
)

omega_history = np.stack(
    [r["omega"] for r in results],
    axis=1,
)


# Reduce the number of frames for animation.
FRAME_STEP = 100

frame_indices = np.unique(
    np.append(
        np.arange(0, len(TIME), FRAME_STEP),
        len(TIME) - 1,
    )
)

animation = animate_robots(
    robots=robots,
    trajectory_history=trajectory_history[frame_indices],
    theta_history=theta_history[frame_indices],
    omega_history=omega_history[frame_indices],
    walls=walls,
    box_size=(CORRIDOR_LENGTH, CORRIDOR_WIDTH),
    interval=30,
    trail_length=300,
    show_trails=True,
    show_angular_velocity=True,
    show_time=True,
    show_grid=True,
    robot_colors=[
        "royalblue",
        "darkorange",
        "forestgreen",
        "crimson",
        "purple",
    ],
    figsize=(12, 5),
)


# ============================================================
# 11. Display
# ============================================================

plt.show()
