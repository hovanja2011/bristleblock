
import numpy as np
import matplotlib.pyplot as plt

from bristlebots.robot import Robot
from bristlebots.geometry import Wall
from bristlebots.contacts import ContactParameters
from bristlebots.simulation import step


# ============================================================
# 1. Geometry and robot parameters
# ============================================================

RADIUS = 1.0
MASS = 1.0

CORRIDOR_WIDTH = 5.0 * RADIUS
CORRIDOR_LENGTH = 250.0 * RADIUS

INITIAL_POSITION = np.array([
    5.0 * RADIUS,
    2.5 * RADIUS,
])

INITIAL_ANGLE_DEG = 20.0
INITIAL_ANGLE = np.deg2rad(INITIAL_ANGLE_DEG)

INITIAL_SPEED = 1.0
INITIAL_VELOCITY = INITIAL_SPEED * np.array([
    np.cos(INITIAL_ANGLE),
    np.sin(INITIAL_ANGLE),
])

PROPULSION_FORCE = 1.0
TRANSLATIONAL_DRAG = 1.0
ROTATIONAL_DRAG = 0.1


# ============================================================
# 2. Contact parameters
# ============================================================

NORMAL_STIFFNESS = 1e4
NORMAL_DAMPING = 20.0

TANGENTIAL_STIFFNESS = 1e4
TANGENTIAL_DAMPING = 0.0

# Keep the same ratio as in the previous experiment.
STATIC_TO_KINETIC_RATIO = 5.0 / 3.0

MU_K_VALUES = [
    0.0,
    0.002,
    0.005,
    0.01,
    0.015,
    0.02,
    0.03,
    0.04,
    0.05,
]


# ============================================================
# 3. Numerical parameters
# ============================================================

DT = 5e-4
TOTAL_TIME = 120.0

TIME = np.arange(0.0, TOTAL_TIME + DT, DT)

# Criteria for detecting characteristic events.
ANGLE_THRESHOLD_DEG = 80.0
SPEED_THRESHOLD = 1e-3

# The speed must remain below the threshold for this duration.
STABILITY_DURATION = 0.5
STABILITY_STEPS = int(STABILITY_DURATION / DT)


# ============================================================
# 4. Corridor geometry
# ============================================================

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
# 5. One simulation
# ============================================================

def run_simulation(mu_k):

    mu_s = STATIC_TO_KINETIC_RATIO * mu_k

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
    )

    contact_parameters = ContactParameters(
        normal_stiffness=NORMAL_STIFFNESS,
        normal_damping=NORMAL_DAMPING,
        mu_static=mu_s,
        mu_kinetic=mu_k,
        tangential_stiffness=TANGENTIAL_STIFFNESS,
        tangential_damping=TANGENTIAL_DAMPING,
    )

    # History arrays
    position_history = np.zeros((len(TIME), 2))
    velocity_history = np.zeros((len(TIME), 2))
    speed_history = np.zeros(len(TIME))
    theta_history = np.zeros(len(TIME))
    omega_history = np.zeros(len(TIME))

    for i, t in enumerate(TIME):

        position_history[i] = robot.position
        velocity_history[i] = robot.velocity
        speed_history[i] = np.linalg.norm(robot.velocity)
        theta_history[i] = robot.theta
        omega_history[i] = robot.omega

        if i == len(TIME) - 1:
            break

        step(
            robot,
            DT,
            walls=walls,
            contact_parameters=contact_parameters,
        )

    return {
        "mu_k": mu_k,
        "mu_s": mu_s,
        "time": TIME.copy(),
        "position": position_history,
        "velocity": velocity_history,
        "speed": speed_history,
        "theta": theta_history,
        "omega": omega_history,
    }


# ============================================================
# 6. Characteristic times
# ============================================================

def first_crossing_time(values, threshold, time):

    indices = np.where(values >= threshold)[0]

    if len(indices) == 0:
        return np.nan

    return time[indices[0]]


def first_stable_low_speed_time(
    speed,
    time,
    threshold,
    window_steps,
):
    """
    Find the first time at which speed remains below threshold
    for at least the specified stability window.
    """

    below = (speed < threshold).astype(int)

    cumulative = np.concatenate([
        [0],
        np.cumsum(below),
    ])

    window_sums = (
        cumulative[window_steps:]
        - cumulative[:-window_steps]
    )

    stable_indices = np.where(
        window_sums == window_steps
    )[0]

    if len(stable_indices) == 0:
        return np.nan

    return time[stable_indices[0]]


# ============================================================
# 7. Run friction sweep
# ============================================================

results = []

for mu_k in MU_K_VALUES:

    print(f"Running mu_k = {mu_k:.4f}")

    result = run_simulation(mu_k)

    theta_deg = np.rad2deg(result["theta"])

    t_80 = first_crossing_time(
        theta_deg,
        ANGLE_THRESHOLD_DEG,
        result["time"],
    )

    t_stop = first_stable_low_speed_time(
        result["speed"],
        result["time"],
        SPEED_THRESHOLD,
        STABILITY_STEPS,
    )

    # Average over the last 10% of the simulation.
    final_start = int(0.9 * len(TIME))

    result["final_angle"] = np.mean(
        theta_deg[final_start:]
    )

    result["final_speed"] = np.mean(
        result["speed"][final_start:]
    )

    result["final_omega"] = np.mean(
        result["omega"][final_start:]
    )

    result["final_y"] = np.mean(
        result["position"][final_start:, 1]
    )

    result["t_80"] = t_80
    result["t_stop"] = t_stop

    results.append(result)


# ============================================================
# 8. Print results
# ============================================================

print("\nCharacteristic times")
print("-" * 75)
print(
    f"{'mu_k':>8} {'t(80 deg)':>14} "
    f"{'t(stop)':>14} {'final angle':>14}"
)
print("-" * 75)

for r in results:

    t80_text = (
        f"{r['t_80']:.3f}"
        if np.isfinite(r["t_80"])
        else "not reached"
    )

    tstop_text = (
        f"{r['t_stop']:.3f}"
        if np.isfinite(r["t_stop"])
        else "not reached"
    )

    print(
        f"{r['mu_k']:8.3f} "
        f"{t80_text:>14} "
        f"{tstop_text:>14} "
        f"{r['final_angle']:14.3f}"
    )


print("\nFinal-state averages")
print("-" * 78)
print(
    f"{'mu_k':>8} {'mu_s':>8} {'Angle':>12} "
    f"{'Speed':>12} {'Omega':>12} {'y / R':>12}"
)
print("-" * 78)

for r in results:

    print(
        f"{r['mu_k']:8.3f} "
        f"{r['mu_s']:8.3f} "
        f"{r['final_angle']:12.3f} "
        f"{r['final_speed']:12.5f} "
        f"{r['final_omega']:12.5f} "
        f"{r['final_y'] / RADIUS:12.5f}"
    )


# ============================================================
# 9. Plot time evolution
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(10, 12))

for r in results:

    label = rf"$\mu_k={r['mu_k']}$"

    axes[0].plot(
        r["time"],
        np.rad2deg(r["theta"]),
        label=label,
    )

    axes[1].plot(
        r["time"],
        r["speed"],
        label=label,
    )

    axes[2].plot(
        r["time"],
        r["omega"],
        label=label,
    )

axes[0].axhline(
    ANGLE_THRESHOLD_DEG,
    linestyle="--",
    color="black",
    alpha=0.5,
)

axes[1].axhline(
    SPEED_THRESHOLD,
    linestyle="--",
    color="black",
    alpha=0.5,
)

axes[0].set_ylabel("Orientation (deg)")
axes[1].set_ylabel("Speed")
axes[2].set_ylabel("Angular velocity")
axes[2].set_xlabel("Time")

for ax in axes:
    ax.grid(True)
    ax.legend(ncol=3)

fig.suptitle("Corridor: friction sweep")
fig.tight_layout()


# ============================================================
# 10. Plot characteristic times
# ============================================================

mu_values = np.array([r["mu_k"] for r in results])

t80_values = np.array([r["t_80"] for r in results])
tstop_values = np.array([r["t_stop"] for r in results])

fig2, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    mu_values,
    t80_values,
    "o-",
    label=r"$t(\theta \geq 80^\circ)$",
)

ax.plot(
    mu_values,
    tstop_values,
    "s-",
    label=r"$t(\mathrm{stop})$",
)

ax.set_xlabel(r"Kinetic friction coefficient $\mu_k$")
ax.set_ylabel("Time")
ax.set_title("Characteristic times vs friction")
ax.grid(True)
ax.legend()
fig2.tight_layout()


# ============================================================
# 11. Plot final state
# ============================================================

final_angles = np.array([
    r["final_angle"] for r in results
])

final_speeds = np.array([
    r["final_speed"] for r in results
])

fig3, axes = plt.subplots(2, 1, figsize=(8, 8))

axes[0].plot(mu_values, final_angles, "o-")
axes[0].set_ylabel("Final orientation (deg)")
axes[0].grid(True)

axes[1].plot(mu_values, final_speeds, "o-")
axes[1].set_xlabel(r"Kinetic friction coefficient $\mu_k$")
axes[1].set_ylabel("Final speed")
axes[1].grid(True)

fig3.suptitle("Final state vs friction")
fig3.tight_layout()

plt.show()
