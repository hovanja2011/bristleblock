
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyArrowPatch


def animate_robots(
    robots,
    trajectory_history,
    omega_history,
    theta_history,
    walls=None,
    box_size=(20, 20),
    interval=30,
    trail_length=100,
):
    """
    Animate robot motion in 2D.

    robots:
        list of Robot objects

    trajectory_history:
        list of arrays of robot positions at each time step

    box_size:
        (width, height) of simulation domain

    interval:
        animation interval in milliseconds

    trail_length:
        number of previous positions displayed
    """

    fig, ax = plt.subplots(figsize=(8, 8))

    width, height = box_size

    ax.set_xlim(-1, width)
    ax.set_ylim(-1, height)
    ax.set_aspect("equal")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Bristle-bot simulation")

    ax.grid(alpha=0.2)

    # Draw walls
    if walls is not None:
        for wall in walls:
            ax.plot(
                [wall.start[0], wall.end[0]],
                [wall.start[1], wall.end[1]],
                color="black",
                linewidth=3,
            )

    # Robot bodies
    circles = []

    # Direction arrows
    arrows = []

    for robot in robots:

        circle = Circle(
            robot.position,
            robot.radius,
            facecolor="royalblue",
            edgecolor="black",
            alpha=0.8,
        )

        ax.add_patch(circle)
        circles.append(circle)

        arrow = FancyArrowPatch(
            robot.position,
            robot.position + robot.radius * robot.direction,
            arrowstyle="-|>",
            mutation_scale=12,
            color="black",
        )

        ax.add_patch(arrow)
        arrows.append(arrow)

    # Trajectories
    trails = []

    for robot in robots:

        line, = ax.plot(
            [],
            [],
            linewidth=1,
            alpha=0.5,
        )

        trails.append(line)

    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
    )

    omega_texts = []

    for robot in robots:
        text = ax.text(
            robot.position[0],
            robot.position[1] + robot.radius + 0.2,
            "",
            ha="center",
            fontsize=9,
        )
        omega_texts.append(text)

    def update(frame):

        positions = trajectory_history[frame]

        for i, robot in enumerate(robots):

            x, y = positions[i]

            # Update body
            circles[i].center = (x, y)

            omega_texts[i].set_position(
                (x, y + robot.radius + 0.2)
            )

            omega_texts[i].set_text(
                f"ω = {omega_history[frame, i]:.2f}"
            )

            # Update direction arrow
            theta = theta_history[frame, i]

            direction = np.array([
                np.cos(theta),
                np.sin(theta),
            ])

            start = np.array([x, y])

            end = (
                start
                + robot.radius * direction
            )

            arrows[i].set_positions(
                start,
                end,
            )

            # Update trajectory
            start_frame = max(
                0,
                frame - trail_length,
            )

            trajectory = np.array(
                trajectory_history[
                    start_frame:frame + 1
                ]
            )

            if len(trajectory) > 0:

                trails[i].set_data(
                    trajectory[:, i, 0],
                    trajectory[:, i, 1],
                )

        time_text.set_text(
            f"Frame: {frame}"
        )

        return (
            circles
            + arrows
            + trails
            + omega_texts
            + [time_text]
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(trajectory_history),
        interval=interval,
        blit=True,
    )

    plt.tight_layout()

    return animation
