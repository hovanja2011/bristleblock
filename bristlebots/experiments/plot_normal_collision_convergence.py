
import numpy as np
import matplotlib.pyplot as plt

from bristlebots.robot import Robot
from bristlebots.geometry import Wall
from bristlebots.contacts import ContactParameters
from bristlebots.simulation import step


# --------------------------------------------------
# Physical parameters
# --------------------------------------------------

MASS = 1.0
RADIUS = 0.5

INITIAL_VX = 20.0
NORMAL_STIFFNESS = 4e5

WALL_POSITION = 1.0


# --------------------------------------------------
# Setup
# --------------------------------------------------

def make_setup():
    """Create a robot initially touching a vertical wall."""

    robot = Robot(
        position=[WALL_POSITION - RADIUS, 0.0],
        velocity=[INITIAL_VX, 0.0],
        radius=RADIUS,
        mass=MASS,
        propulsion_force=0.0,
        translational_drag=0.0,
        rotational_drag=0.0,
    )

    wall = Wall(
        start=[WALL_POSITION, -1.0],
        end=[WALL_POSITION, 5.0],
    )

    parameters = ContactParameters(
        normal_stiffness=NORMAL_STIFFNESS,
        normal_damping=0.0,
        mu_static=0.0,
        mu_kinetic=0.0,
        tangential_stiffness=1e4,
        tangential_damping=0.0,
    )

    return robot, wall, parameters


# --------------------------------------------------
# Analytical solution
# --------------------------------------------------

def analytical_solution(time):

    omega = np.sqrt(NORMAL_STIFFNESS / MASS)

    delta = INITIAL_VX / omega * np.sin(omega * time)
    vx = INITIAL_VX * np.cos(omega * time)

    return delta, vx


# --------------------------------------------------
# Numerical solution
# --------------------------------------------------

def numerical_solution(dt, total_time):

    robot, wall, parameters = make_setup()

    steps = round(total_time / dt)

    time = np.arange(steps + 1) * dt

    delta = [0.0]
    vx = [robot.velocity[0]]

    for _ in range(steps):

        step(
            robot,
            dt,
            walls=[wall],
            contact_parameters=parameters,
        )

        penetration = max(
            robot.position[0] + RADIUS - WALL_POSITION,
            0.0,
        )

        delta.append(penetration)
        vx.append(robot.velocity[0])

    return time, np.array(delta), np.array(vx)


# --------------------------------------------------
# Error calculation
# --------------------------------------------------

def calculate_errors(dt):

    omega = np.sqrt(NORMAL_STIFFNESS / MASS)
    contact_time = np.pi / omega

    total_time = 0.9 * contact_time

    time, delta_num, vx_num = numerical_solution(
        dt,
        total_time,
    )

    delta_exact, vx_exact = analytical_solution(time)

    error_delta = np.max(np.abs(delta_num - delta_exact))
    error_vx = np.max(np.abs(vx_num - vx_exact))

    return error_delta, error_vx


# --------------------------------------------------
# Convergence analysis
# --------------------------------------------------

def main():

    omega = np.sqrt(NORMAL_STIFFNESS / MASS)
    contact_time = np.pi / omega

    # Coarse to fine time steps
    time_steps = contact_time / np.array([
        100,
        200,
        400,
        800,
    ])

    errors_delta = []
    errors_vx = []

    for dt in time_steps:

        error_delta, error_vx = calculate_errors(dt)

        errors_delta.append(error_delta)
        errors_vx.append(error_vx)

    errors_delta = np.array(errors_delta)
    errors_vx = np.array(errors_vx)

    # Observed convergence orders
    order_delta = (
        np.log(errors_delta[:-1] / errors_delta[1:])
        / np.log(2.0)
    )

    order_vx = (
        np.log(errors_vx[:-1] / errors_vx[1:])
        / np.log(2.0)
    )

    # --------------------------------------------------
    # Print numerical results
    # --------------------------------------------------

    print("\nNormal collision: convergence analysis\n")

    print(
        f"{'dt':>12} "
        f"{'Error delta':>16} "
        f"{'Order delta':>14} "
        f"{'Error vx':>16} "
        f"{'Order vx':>12}"
    )

    print("-" * 76)

    for i, dt in enumerate(time_steps):

        p_delta = "-" if i == 0 else f"{order_delta[i-1]:.4f}"
        p_vx = "-" if i == 0 else f"{order_vx[i-1]:.4f}"

        print(
            f"{dt:12.6e} "
            f"{errors_delta[i]:16.6e} "
            f"{p_delta:>14} "
            f"{errors_vx[i]:16.6e} "
            f"{p_vx:>12}"
        )

    # Global fitted slopes
    slope_delta = np.polyfit(
        np.log(time_steps),
        np.log(errors_delta),
        1,
    )[0]

    slope_vx = np.polyfit(
        np.log(time_steps),
        np.log(errors_vx),
        1,
    )[0]

    print("\nFitted convergence slopes:")
    print(f"Coordinate delta: {slope_delta:.4f}")
    print(f"Velocity vx:      {slope_vx:.4f}")

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    quantities = [
        (errors_delta, 2, r"Maximum error in $\delta$"),
        (errors_vx, 1, r"Maximum error in $v_x$"),
    ]

    for ax, (errors, expected_order, ylabel) in zip(
        axes,
        quantities,
    ):

        ax.loglog(
            time_steps,
            errors,
            "o-",
            label="Numerical error",
        )

        reference = (
            errors[-1]
            * (time_steps / time_steps[-1]) ** expected_order
        )

        ax.loglog(
            time_steps,
            reference,
            "--",
            label=fr"Reference: $O(dt^{expected_order})$",
        )

        ax.set_xlabel(r"Time step $dt$")
        ax.set_ylabel(ylabel)

        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.suptitle("Normal collision: convergence analysis")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
