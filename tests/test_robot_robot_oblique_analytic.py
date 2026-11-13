
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Circle

from bristlebots.robot import Robot
from bristlebots.contacts import ContactParameters
from bristlebots.simulation import step_robots


def cross_2d(a, b):
    return a[0] * b[1] - a[1] * b[0]


def total_momentum(robots):
    return sum(
        (robot.mass * robot.velocity for robot in robots),
        start=np.zeros(2),
    )


def kinetic_energy(robots):
    energy = 0.0

    for robot in robots:
        energy += 0.5 * robot.mass * np.dot(
            robot.velocity, robot.velocity
        )
        energy += 0.5 * robot.moment_of_inertia * robot.omega**2

    return energy


def angular_momentum(robots):
    result = 0.0

    for robot in robots:
        orbital = robot.mass * cross_2d(
            robot.position, robot.velocity
        )
        spin = robot.moment_of_inertia * robot.omega

        result += orbital + spin

    return result


def plot_collision(
    times,
    positions,
    velocities,
    momentum_history,
    kinetic_history,
    potential_history,
    angular_history,
    maximum_overlap_index,
    radius,
    normal,
    velocity_a_initial,
    velocity_b_initial,
    velocity_a_expected,
    velocity_b_expected,
    momentum_initial,
    energy_initial,
    angular_initial,
):
    """Create diagnostic plots for the elastic collision test."""

    # --------------------------------------------------
    # Figure 1: trajectories and collision geometry
    # --------------------------------------------------

    fig1, ax = plt.subplots(figsize=(9, 7))

    ax.plot(
        positions[:, 0, 0],
        positions[:, 0, 1],
        label="Robot A trajectory",
    )

    ax.plot(
        positions[:, 1, 0],
        positions[:, 1, 1],
        label="Robot B trajectory",
    )

    # Initial and final positions
    for i, label in enumerate(["A", "B"]):
        ax.scatter(
            positions[0, i, 0],
            positions[0, i, 1],
            marker="o",
            s=55,
            label=f"Robot {label}: initial",
        )

        ax.scatter(
            positions[-1, i, 0],
            positions[-1, i, 1],
            marker="x",
            s=65,
            label=f"Robot {label}: final",
        )

    # Draw robot bodies at maximum overlap
    collision_positions = positions[maximum_overlap_index]

    for i in range(2):
        circle = Circle(
            collision_positions[i],
            radius,
            fill=False,
            linestyle="--",
            linewidth=1.5,
        )
        ax.add_patch(circle)

    # Initial and expected final velocity arrows
    arrow_scale = 0.35

    for i, (initial_velocity, final_velocity) in enumerate([
        (velocity_a_initial, velocity_a_expected),
        (velocity_b_initial, velocity_b_expected),
    ]):
        initial_position = positions[0, i]
        final_position = positions[-1, i]

        ax.quiver(
            initial_position[0],
            initial_position[1],
            initial_velocity[0],
            initial_velocity[1],
            angles="xy",
            scale_units="xy",
            scale=1 / arrow_scale,
            color="black",
            width=0.006,
        )

        ax.quiver(
            final_position[0],
            final_position[1],
            final_velocity[0],
            final_velocity[1],
            angles="xy",
            scale_units="xy",
            scale=1 / arrow_scale,
            color="green",
            width=0.006,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Oblique elastic collision: trajectories")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # --------------------------------------------------
    # Figure 2: velocity components
    # --------------------------------------------------

    fig2, axes = plt.subplots(
        2, 1,
        figsize=(10, 7),
        sharex=True,
    )

    velocity_labels = [
        ("x velocity", 0),
        ("y velocity", 1),
    ]

    initial_velocities = [
        velocity_a_initial,
        velocity_b_initial,
    ]

    expected_velocities = [
        velocity_a_expected,
        velocity_b_expected,
    ]

    for ax, (component_name, component) in zip(
        axes, velocity_labels
    ):
        ax.plot(
            times,
            velocities[:, 0, component],
            label="Robot A: numerical",
        )

        ax.plot(
            times,
            velocities[:, 1, component],
            label="Robot B: numerical",
        )

        # Analytical initial and final values.
        # The finite-duration spring collision is not instantaneous,
        # so these are reference levels, not a time-resolved solution.
        for i, color in enumerate(["C0", "C1"]):
            ax.axhline(
                initial_velocities[i][component],
                color=color,
                linestyle=":",
                alpha=0.7,
                label=f"Robot {'A' if i == 0 else 'B'}: initial",
            )

            ax.axhline(
                expected_velocities[i][component],
                color=color,
                linestyle="--",
                alpha=0.7,
                label=f"Robot {'A' if i == 0 else 'B'}: analytical final",
            )

        ax.set_ylabel(component_name)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

    axes[-1].set_xlabel("Time")
    fig2.suptitle("Velocity components during collision")

    # --------------------------------------------------
    # Figure 3: conservation laws
    # --------------------------------------------------

    fig3, axes = plt.subplots(
        2, 2,
        figsize=(11, 8),
        sharex=True,
    )

    px_error = momentum_history[:, 0] - momentum_initial[0]
    py_error = momentum_history[:, 1] - momentum_initial[1]

    mechanical_energy = kinetic_history + potential_history
    energy_error = mechanical_energy - energy_initial

    angular_error = angular_history - angular_initial

    axes[0, 0].plot(times, px_error, label="ΔPx")
    axes[0, 0].plot(times, py_error, label="ΔPy")
    axes[0, 0].set_title("Linear momentum")
    axes[0, 0].set_ylabel("Change")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(
        times,
        kinetic_history,
        label="Kinetic energy",
    )
    axes[0, 1].plot(
        times,
        potential_history,
        label="Contact spring energy",
    )
    axes[0, 1].plot(
        times,
        mechanical_energy,
        linestyle="--",
        label="Total mechanical energy",
    )
    axes[0, 1].set_title("Energy")
    axes[0, 1].set_ylabel("Energy")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(times, energy_error)
    axes[1, 0].set_title("Mechanical energy error")
    axes[1, 0].set_ylabel("E - E_initial")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(times, angular_error)
    axes[1, 1].set_title("Angular momentum error")
    axes[1, 1].set_ylabel("Lz - Lz_initial")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].grid(alpha=0.3)

    fig3.suptitle("Conservation diagnostics")

    plt.tight_layout()
    plt.show()


def test_oblique_collision_against_elastic_analytic_solution():

    # --------------------------------------------------
    # Parameters
    # --------------------------------------------------

    R = 1.0
    mass = 1.0

    dt = 1e-5
    total_time = 0.15
    steps = int(total_time / dt)

    kn = 2000.0

    contact_parameters = ContactParameters(
        normal_stiffness=kn,
        normal_damping=0.0,
        mu_static=0.0,
        mu_kinetic=0.0,
        tangential_stiffness=500.0,
        tangential_damping=0.0,
    )

    # --------------------------------------------------
    # Initial collision geometry
    # --------------------------------------------------

    angle = np.deg2rad(35.0)

    normal = np.array([
        np.cos(angle),
        np.sin(angle),
    ])

    center_velocity = np.array([0.2, -0.1])
    relative_speed = 1.0

    initial_distance = 2.0 * R + 0.05

    position_a = np.array([0.0, 0.0])
    position_b = initial_distance * normal

    velocity_a_initial = (
        center_velocity + 0.5 * relative_speed * normal
    )

    velocity_b_initial = (
        center_velocity - 0.5 * relative_speed * normal
    )

    robot_a = Robot(
        position=position_a,
        velocity=velocity_a_initial,
        theta=0.0,
        omega=0.7,
        radius=R,
        mass=mass,
        propulsion_force=0.0,
        translational_drag=0.0,
        rotational_drag=0.0,
    )

    robot_b = Robot(
        position=position_b,
        velocity=velocity_b_initial,
        theta=0.0,
        omega=-0.4,
        radius=R,
        mass=mass,
        propulsion_force=0.0,
        translational_drag=0.0,
        rotational_drag=0.0,
    )

    robots = [robot_a, robot_b]

    # --------------------------------------------------
    # Initial conserved quantities
    # --------------------------------------------------

    momentum_initial = total_momentum(robots)
    energy_initial = kinetic_energy(robots)
    angular_initial = angular_momentum(robots)

    # --------------------------------------------------
    # Analytical final velocities
    # --------------------------------------------------

    velocity_a_expected = (
        center_velocity - 0.5 * relative_speed * normal
    )

    velocity_b_expected = (
        center_velocity + 0.5 * relative_speed * normal
    )

    # --------------------------------------------------
    # History arrays
    # --------------------------------------------------

    times = [0.0]

    positions = [[robot.position.copy() for robot in robots]]
    velocities = [[robot.velocity.copy() for robot in robots]]

    momentum_history = [momentum_initial.copy()]
    kinetic_history = [energy_initial]
    potential_history = [0.0]
    angular_history = [angular_initial]

    maximum_overlap = 0.0
    maximum_overlap_index = 0

    contact_detected = False

    # --------------------------------------------------
    # Simulation
    # --------------------------------------------------

    for step in range(steps):

        step_robots(
            robots,
            dt,
            contact_parameters=contact_parameters,
        )

        distance = np.linalg.norm(
            robot_b.position - robot_a.position
        )

        overlap = max(0.0, 2.0 * R - distance)

        if overlap > 0:
            contact_detected = True

        if overlap > maximum_overlap:
            maximum_overlap = overlap
            maximum_overlap_index = step + 1

        time = (step + 1) * dt

        times.append(time)

        positions.append([
            robot.position.copy() for robot in robots
        ])

        velocities.append([
            robot.velocity.copy() for robot in robots
        ])

        momentum_history.append(total_momentum(robots))
        kinetic_history.append(kinetic_energy(robots))

        # Elastic potential energy of the normal contact spring.
        potential_history.append(0.5 * kn * overlap**2)

        angular_history.append(angular_momentum(robots))

    # --------------------------------------------------
    # Convert histories to arrays
    # --------------------------------------------------

    times = np.asarray(times)
    positions = np.asarray(positions)
    velocities = np.asarray(velocities)

    momentum_history = np.asarray(momentum_history)
    kinetic_history = np.asarray(kinetic_history)
    potential_history = np.asarray(potential_history)
    angular_history = np.asarray(angular_history)

    # --------------------------------------------------
    # Final values and diagnostics
    # --------------------------------------------------

    momentum_final = total_momentum(robots)
    energy_final = kinetic_energy(robots)
    angular_final = angular_momentum(robots)

    print("\nOblique elastic collision")
    print("-------------------------")

    print("Contact detected:", contact_detected)
    print(f"Maximum overlap: {maximum_overlap:.8f}")
    print(f"Maximum relative overlap: {maximum_overlap / R:.4%}")

    print("\nRobot A:")
    print("Initial velocity:   ", velocity_a_initial)
    print("Numerical final:    ", robot_a.velocity)
    print("Analytical final:   ", velocity_a_expected)

    print("\nRobot B:")
    print("Initial velocity:   ", velocity_b_initial)
    print("Numerical final:    ", robot_b.velocity)
    print("Analytical final:   ", velocity_b_expected)

    print("\nConservation errors:")
    print(
        "Linear momentum:",
        np.linalg.norm(momentum_final - momentum_initial),
    )
    print(
        "Kinetic energy:",
        abs(energy_final - energy_initial),
    )
    print(
        "Mechanical energy:",
        abs(
            kinetic_history[-1]
            + potential_history[-1]
            - energy_initial
        ),
    )
    print(
        "Angular momentum:",
        abs(angular_final - angular_initial),
    )

    print("\nFinal velocity errors:")
    print(
        "Robot A:",
        np.linalg.norm(robot_a.velocity - velocity_a_expected),
    )
    print(
        "Robot B:",
        np.linalg.norm(robot_b.velocity - velocity_b_expected),
    )

    # --------------------------------------------------
    # Plots
    # --------------------------------------------------

    plot_collision(
        times=times,
        positions=positions,
        velocities=velocities,
        momentum_history=momentum_history,
        kinetic_history=kinetic_history,
        potential_history=potential_history,
        angular_history=angular_history,
        maximum_overlap_index=maximum_overlap_index,
        radius=R,
        normal=normal,
        velocity_a_initial=velocity_a_initial,
        velocity_b_initial=velocity_b_initial,
        velocity_a_expected=velocity_a_expected,
        velocity_b_expected=velocity_b_expected,
        momentum_initial=momentum_initial,
        energy_initial=energy_initial,
        angular_initial=angular_initial,
    )

    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------

    assert contact_detected

    np.testing.assert_allclose(
        robot_a.velocity,
        velocity_a_expected,
        atol=2e-3,
        rtol=0,
    )

    np.testing.assert_allclose(
        robot_b.velocity,
        velocity_b_expected,
        atol=2e-3,
        rtol=0,
    )

    np.testing.assert_allclose(
        momentum_final,
        momentum_initial,
        atol=1e-9,
        rtol=0,
    )

    np.testing.assert_allclose(
        energy_final,
        energy_initial,
        atol=2e-3,
        rtol=0,
    )

    np.testing.assert_allclose(
        angular_final,
        angular_initial,
        atol=2e-3,
        rtol=0,
    )

    # With zero friction, the angular velocities must not change.
    assert abs(robot_a.omega - 0.7) < 1e-10
    assert abs(robot_b.omega + 0.4) < 1e-10
