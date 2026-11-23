"""Plotting and animation helpers for continuum density and velocity fields."""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

from toner_tu.fields import Grid

from toner_tu.operators import vorticity


def calculate_vorticity_history(
    vx_history,
    vy_history,
    grid,
):
    """Calculate vorticity for every saved velocity field."""

    return np.array([
        vorticity(vx, vy, grid)
        for vx, vy in zip(vx_history, vy_history)
    ])


def plot_diagnostics(
    times,
    polarizations,
    speeds,
    min_densities,
):
    """Plot global diagnostics as functions of time."""

    fig, axes = plt.subplots(
        3, 1,
        figsize=(9, 9),
        sharex=True,
    )

    axes[0].plot(times, polarizations)
    axes[0].set_ylabel("Polarization P")

    axes[1].plot(times, speeds)
    axes[1].set_ylabel("Mean speed")

    axes[2].plot(times, min_densities)
    axes[2].set_ylabel("Minimum density")
    axes[2].set_xlabel("Time")

    for ax in axes:
        ax.grid(True)

    fig.tight_layout()
    return fig, axes


def animate_fields(
    grid: Grid,
    times,
    rho_history,
    vx_history,
    vy_history,
    skip: int = 4,
    interval: int = 100,
):
    """
    Animate density and velocity fields.

    Density is shown as a heatmap.
    Velocity is shown as arrows at fixed grid positions.
    Arrow directions and magnitudes are updated at every frame.
    """

    rho_history = np.asarray(rho_history)
    vx_history = np.asarray(vx_history)
    vy_history = np.asarray(vy_history)

    expected_shape = (len(times), grid.ny, grid.nx)

    if rho_history.shape != expected_shape:
        raise ValueError(
            f"Expected rho_history shape {expected_shape}, "
            f"got {rho_history.shape}."
        )

    if vx_history.shape != expected_shape:
        raise ValueError(
            f"Expected vx_history shape {expected_shape}, "
            f"got {vx_history.shape}."
        )

    if vy_history.shape != expected_shape:
        raise ValueError(
            f"Expected vy_history shape {expected_shape}, "
            f"got {vy_history.shape}."
        )

    if skip < 1:
        raise ValueError("skip must be at least 1.")

    x, y = grid.coordinates()

    fig, ax = plt.subplots(figsize=(8, 8))

    image = ax.imshow(
        rho_history[0],
        origin="lower",
        extent=[
            -grid.dx / 2,
            grid.lx - grid.dx / 2,
            -grid.dy / 2,
            grid.ly - grid.dy / 2,
        ],
        cmap="viridis",
        vmin=np.min(rho_history),
        vmax=np.max(rho_history),
        interpolation="nearest",
        animated=False,
        aspect="equal",
    )

    # Fixed spatial positions of the arrows.
    arrow_x = x[::skip, ::skip]
    arrow_y = y[::skip, ::skip]

    quiver = ax.quiver(
        arrow_x,
        arrow_y,
        vx_history[0, ::skip, ::skip],
        vy_history[0, ::skip, ::skip],
        color="white",
        pivot="middle",
        angles="xy",
        scale_units="xy",
        scale=None,
    )

    title = ax.set_title(f"t = {times[0]:.3f}")

    ax.set_xlim(0, grid.lx)
    ax.set_ylim(0, grid.ly)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    fig.colorbar(image, ax=ax, label="Density")

    def update(frame):
        image.set_data(rho_history[frame])

        # Update velocity vectors at the current time.
        quiver.set_UVC(
            vx_history[frame, ::skip, ::skip],
            vy_history[frame, ::skip, ::skip],
        )

        title.set_text(f"t = {times[frame]:.3f}")

        return image, quiver, title

    animation = FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=interval,
        blit=False,
        repeat=True,
        cache_frame_data=False,
    )

    return fig, animation


def animate_fields_with_vorticity(
    grid: Grid,
    times,
    rho_history,
    vx_history,
    vy_history,
    skip: int = 4,
    interval: int = 100,
):
    """
    Animate density, velocity and vorticity.

    Left: density and velocity vectors.
    Right: vorticity field.
    """

    rho_history = np.asarray(rho_history)
    vx_history = np.asarray(vx_history)
    vy_history = np.asarray(vy_history)

    omega_history = calculate_vorticity_history(
        vx_history,
        vy_history,
        grid,
    )

    x, y = grid.coordinates()

    fig, axes = plt.subplots(
        1, 2,
        figsize=(15, 7),
    )

    ax1, ax2 = axes

    # Shared density limits
    rho_min = np.min(rho_history)
    rho_max = np.max(rho_history)

    # Symmetric vorticity limits
    omega_max = np.max(np.abs(omega_history))

    if omega_max == 0:
        omega_max = 1.0

    # ----------------------------------------------
    # Left panel: density + velocity
    # ----------------------------------------------

    image_rho = ax1.imshow(
        rho_history[0],
        origin="lower",
        extent=[
            -grid.dx / 2,
            grid.lx - grid.dx / 2,
            -grid.dy / 2,
            grid.ly - grid.dy / 2,
        ],
        cmap="viridis",
        vmin=rho_min,
        vmax=rho_max,
        interpolation="nearest",
    )

    arrow_x = x[::skip, ::skip]
    arrow_y = y[::skip, ::skip]

    quiver = ax1.quiver(
        arrow_x,
        arrow_y,
        vx_history[0, ::skip, ::skip],
        vy_history[0, ::skip, ::skip],
        color="white",
        pivot="middle",
        angles="xy",
        scale_units="xy",
    )

    ax1.set_title("Density and velocity")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    fig.colorbar(
        image_rho,
        ax=ax1,
        label="Density",
    )

    # ----------------------------------------------
    # Right panel: vorticity
    # ----------------------------------------------

    image_omega = ax2.imshow(
        omega_history[0],
        origin="lower",
        extent=[
            -grid.dx / 2,
            grid.lx - grid.dx / 2,
            -grid.dy / 2,
            grid.ly - grid.dy / 2,
        ],
        cmap="RdBu_r",
        vmin=-omega_max,
        vmax=omega_max,
        interpolation="nearest",
    )

    ax2.set_title("Vorticity")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    fig.colorbar(
        image_omega,
        ax=ax2,
        label="Vorticity",
    )

    title = fig.suptitle(f"t = {times[0]:.3f}")

    for ax in axes:
        ax.set_xlim(0, grid.lx)
        ax.set_ylim(0, grid.ly)
        ax.set_aspect("equal")

    # ----------------------------------------------
    # Animation update
    # ----------------------------------------------

    def update(frame):
        image_rho.set_data(rho_history[frame])

        quiver.set_UVC(
            vx_history[frame, ::skip, ::skip],
            vy_history[frame, ::skip, ::skip],
        )

        image_omega.set_data(omega_history[frame])

        title.set_text(f"t = {times[frame]:.3f}")

        return image_rho, quiver, image_omega, title

    animation = FuncAnimation(
        fig,
        update,
        frames=len(times),
        interval=interval,
        blit=False,
        repeat=True,
        cache_frame_data=False,
    )

    fig.tight_layout()

    return fig, animation
