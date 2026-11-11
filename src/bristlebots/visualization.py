
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyArrowPatch


def animate_robots(
    robots,
    trajectory_history,
    omega_history=None,
    theta_history=None,
    walls=None,
    box_size=(20, 20),
    interval=30,
    trail_length=100,
    show_trails=True,
    show_angular_velocity=True,
    show_time=True,
    show_grid=True,
    robot_colors=None,
    figsize=(8, 8),
):
    """
    Animate the motion of bristlebot-like robots in 2D.

    Parameters
    ----------
    robots : list
        List of Robot objects.

    trajectory_history : array-like
        Robot positions at each time step.
        Expected shape: (n_frames, n_robots, 2).

    omega_history : array-like or None
        Angular velocities.
        Expected shape: (n_frames, n_robots).
        If None, current robot angular velocities are used.

    theta_history : array-like or None
        Robot orientations.
        Expected shape: (n_frames, n_robots).
        If None, current robot orientations are used.

    walls : list or None
        List of Wall objects.

    box_size : tuple
        Width and height of the simulation domain.

    interval : int
        Delay between animation frames in milliseconds.

    trail_length : int
        Number of previous positions shown in trajectories.

    show_trails : bool
        Whether to display robot trajectories.

    show_angular_velocity : bool
        Whether to display angular velocity values.

    show_time : bool
        Whether to display the current frame number.

    show_grid : bool
        Whether to display the coordinate grid.

    robot_colors : list or None
        List of colors, one for each robot.

    figsize : tuple
        Figure size in inches.

    Returns
    -------
    animation : matplotlib.animation.FuncAnimation
        Animation object.
    """

    # --------------------------------------------------
    # Validate input data
    # --------------------------------------------------

    trajectory_history = np.asarray(
        trajectory_history,
        dtype=float,
    )

    n_frames = len(trajectory_history)
    n_robots = len(robots)

    if trajectory_history.shape != (n_frames, n_robots, 2):
        raise ValueError(
            "trajectory_history must have shape "
            "(n_frames, n_robots, 2)."
        )

    if n_frames == 0:
        raise ValueError("trajectory_history cannot be empty.")

    if interval <= 0:
        raise ValueError("interval must be positive.")

    if trail_length <= 0:
        raise ValueError("trail_length must be positive.")

    width, height = box_size

    if width <= 0 or height <= 0:
        raise ValueError("box_size dimensions must be positive.")

    # --------------------------------------------------
    # Prepare orientation and angular velocity histories
    # --------------------------------------------------

    if theta_history is None:
        theta_history = np.tile(
            [robot.theta for robot in robots],
            (n_frames, 1),
        )
    else:
        theta_history = np.asarray(
            theta_history,
            dtype=float,
        )

    if omega_history is None:
        omega_history = np.tile(
            [robot.omega for robot in robots],
            (n_frames, 1),
        )
    else:
        omega_history = np.asarray(
            omega_history,
            dtype=float,
        )

    expected_shape = (n_frames, n_robots)

    if theta_history.shape != expected_shape:
        raise ValueError(
            "theta_history must have shape "
            "(n_frames, n_robots)."
        )

    if omega_history.shape != expected_shape:
        raise ValueError(
            "omega_history must have shape "
            "(n_frames, n_robots)."
        )

    # --------------------------------------------------
    # Prepare robot colors
    # --------------------------------------------------

    if robot_colors is None:
        robot_colors = ["royalblue"] * n_robots

    if len(robot_colors) != n_robots:
        raise ValueError(
            "robot_colors must contain one color per robot."
        )

    # --------------------------------------------------
    # Create figure and axes
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlim(-1, width+1)
    ax.set_ylim(-1, height+1)

    ax.set_aspect("equal")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Bristlebot simulation")

    if show_grid:
        ax.grid(alpha=0.2)

    # --------------------------------------------------
    # Draw walls
    # --------------------------------------------------

    if walls is not None:

        for wall in walls:

            ax.plot(
                [wall.start[0], wall.end[0]],
                [wall.start[1], wall.end[1]],
                color="black",
                linewidth=3,
            )

    # --------------------------------------------------
    # Create robot bodies and direction arrows
    # --------------------------------------------------

    circles = []
    arrows = []

    for i, robot in enumerate(robots):

        circle = Circle(
            trajectory_history[0, i],
            robot.radius,
            facecolor=robot_colors[i],
            edgecolor="black",
            alpha=0.8,
        )

        ax.add_patch(circle)
        circles.append(circle)

        initial_position = trajectory_history[0, i]

        initial_direction = np.array([
            np.cos(theta_history[0, i]),
            np.sin(theta_history[0, i]),
        ])

        arrow = FancyArrowPatch(
            initial_position,
            initial_position + robot.radius * initial_direction,
            arrowstyle="-|>",
            mutation_scale=12,
            color="black",
        )

        ax.add_patch(arrow)
        arrows.append(arrow)

    # --------------------------------------------------
    # Create trajectory lines
    # --------------------------------------------------

    trails = []

    for i in range(n_robots):

        line, = ax.plot(
            [],
            [],
            color=robot_colors[i],
            linewidth=1,
            alpha=0.5,
        )

        trails.append(line)


    # --------------------------------------------------
    # Drive point legend
    # --------------------------------------------------

    legend_labels = []

    for i, robot in enumerate(robots):

        offset = np.asarray(
            getattr(robot, "propulsion_offset", [0.0, 0.0]),
            dtype=float,
        )

        b = offset[1]

        legend_labels.append(
            f"Robot {i + 1}: b/R = {b / robot.radius:.3f}"
        )

    ax.legend(
        trails,
        legend_labels,
        loc="upper right",
        fontsize=9,
        title="Propulsion point",
    )
    # --------------------------------------------------
    # Create angular velocity labels
    # --------------------------------------------------

    omega_texts = []

    for i, robot in enumerate(robots):

        text = ax.text(
            trajectory_history[0, i, 0],
            trajectory_history[0, i, 1] + robot.radius + 0.2,
            "",
            ha="center",
            fontsize=9,
        )

        omega_texts.append(text)

    # --------------------------------------------------
    # Time label
    # --------------------------------------------------

    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
    )

    # --------------------------------------------------
    # Animation update function
    # --------------------------------------------------

    def update(frame):

        positions = trajectory_history[frame]

        for i, robot in enumerate(robots):

            x, y = positions[i]

            # Update robot body
            circles[i].center = (x, y)

            # Update direction arrow
            theta = theta_history[frame, i]

            direction = np.array([
                np.cos(theta),
                np.sin(theta),
            ])

            start = np.array([x, y])
            end = start + robot.radius * direction

            arrows[i].set_positions(start, end)

            # Update angular velocity label
            omega_texts[i].set_position(
                (x, y + robot.radius + 0.2)
            )

            if show_angular_velocity:
                omega_texts[i].set_text(
                    f"ω = {omega_history[frame, i]:.2f}"
                )
            else:
                omega_texts[i].set_text("")

            # Update trajectory
            if show_trails:

                start_frame = max(
                    0,
                    frame - trail_length + 1,
                )

                trajectory = trajectory_history[
                    start_frame:frame + 1,
                    i,
                ]

                trails[i].set_data(
                    trajectory[:, 0],
                    trajectory[:, 1],
                )

            else:
                trails[i].set_data([], [])

        # Update time label
        if show_time:
            time_text.set_text(f"Frame: {frame}")
        else:
            time_text.set_text("")

        return (
            circles
            + arrows
            + trails
            + omega_texts
            + [time_text]
        )

    # --------------------------------------------------
    # Create animation
    # --------------------------------------------------

    animation = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval,
        blit=False,
    )

    plt.tight_layout()

    return animation
