
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from toner_tu.fields import Grid
from toner_tu.visualization import animate_fields


def load_data(path):
    data = np.load(path)

    times = data["times"]
    rho = data["rho"]
    vx = data["vx"]
    vy = data["vy"]
    x = data["x"]
    y = data["y"]

    return times, rho, vx, vy, x, y


def calculate_diagnostics(times, rho, vx, vy, dx, dy):
    """Calculate density-weighted collective diagnostics."""

    n_frames = len(times)

    total_mass = np.zeros(n_frames)
    mean_density = np.zeros(n_frames)
    mean_speed = np.zeros(n_frames)
    mean_vx = np.zeros(n_frames)
    mean_vy = np.zeros(n_frames)
    polarization = np.zeros(n_frames)
    center_x = np.zeros(n_frames)
    center_y = np.zeros(n_frames)
    max_density = np.zeros(n_frames)
    min_density = np.zeros(n_frames)

    cell_area = dx * dy

    ny, nx = rho.shape[1:]
    x_grid = (np.arange(nx) + 0.5) * dx
    y_grid = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(x_grid, y_grid)

    for i in range(n_frames):
        r = rho[i]
        ux = vx[i]
        uy = vy[i]

        speed = np.sqrt(ux**2 + uy**2)

        mass = np.sum(r) * cell_area
        total_mass[i] = mass
        mean_density[i] = np.mean(r)
        max_density[i] = np.max(r)
        min_density[i] = np.min(r)

        if mass > 1e-12:
            mean_vx[i] = np.sum(r * ux) / np.sum(r)
            mean_vy[i] = np.sum(r * uy) / np.sum(r)
            mean_speed[i] = np.sum(r * speed) / np.sum(r)

            momentum = np.sqrt(
                np.sum(r * ux)**2 + np.sum(r * uy)**2
            )

            total_motion = np.sum(r * speed)

            if total_motion > 1e-12:
                polarization[i] = momentum / total_motion

            ny, nx = r.shape
            x_grid = np.arange(nx) * dx
            y_grid = np.arange(ny) * dy
            X, Y = np.meshgrid(x_grid, y_grid)

            center_x[i] = np.sum(r * X) / np.sum(r)
            center_y[i] = np.sum(r * Y) / np.sum(r)

    return {
        "total_mass": total_mass,
        "mean_density": mean_density,
        "mean_speed": mean_speed,
        "mean_vx": mean_vx,
        "mean_vy": mean_vy,
        "polarization": polarization,
        "center_x": center_x,
        "center_y": center_y,
        "max_density": max_density,
        "min_density": min_density,
    }


def plot_snapshots(times, rho, vx, vy, x, y):
    """Show density and velocity at three moments."""

    indices = [0, len(times) // 2, len(times) - 1]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(16, 5),
        constrained_layout=True,
    )

    X, Y = np.meshgrid(x, y)

    for ax, i in zip(axes, indices):
        image = ax.imshow(
            rho[i],
            origin="lower",
            extent=[
                x[0] - (x[1] - x[0]) / 2,
                x[-1] + (x[1] - x[0]) / 2,
                y[0] - (y[1] - y[0]) / 2,
                y[-1] + (y[1] - y[0]) / 2,
            ],
            aspect="auto",
            cmap="viridis",
        )

        skip = 5
        ax.quiver(
            X[::skip, ::skip],
            Y[::skip, ::skip],
            vx[i, ::skip, ::skip],
            vy[i, ::skip, ::skip],
            color="white",
            scale=20,
        )

        ax.set_title(f"t = {times[i]:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.colorbar(image, ax=axes, label="Density")
    fig.suptitle("Density and velocity field")

    return fig


def plot_diagnostics(times, diagnostics):
    """Plot collective quantities as functions of time."""

    fig, axes = plt.subplots(
        3, 2,
        figsize=(12, 10),
        constrained_layout=True,
    )

    plots = [
        ("total_mass", "Total mass"),
        ("mean_density", "Mean density"),
        ("mean_speed", "Mean speed"),
        ("polarization", "Polarization"),
        ("center_x", "Center of mass: x"),
        ("max_density", "Maximum density"),
    ]

    for ax, (key, label) in zip(axes.flat, plots):
        ax.plot(times, diagnostics[key])
        ax.set_title(label)
        ax.set_xlabel("Time")
        ax.grid(True)

    return fig


def plot_density_profile(times, rho, x):
    """Plot density averaged across the channel width."""

    density_profile = np.mean(rho, axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))

    indices = [
        0,
        len(times) // 4,
        len(times) // 2,
        3 * len(times) // 4,
        len(times) - 1,
    ]

    for i in indices:
        ax.plot(
            x,
            density_profile[i],
            label=f"t = {times[i]:.2f}",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("Density averaged over y")
    ax.set_title("Longitudinal density profile")
    ax.legend()
    ax.grid(True)

    return fig


def main():
    data_path = (
        Path(__file__).parent / "channel_history.npz"
    )

    times, rho, vx, vy, x, y = load_data(data_path)

    ny, nx = rho.shape[1:]

    lx = nx * (x[1] - x[0])
    ly = ny * (y[1] - y[0])

    grid = Grid(nx=nx, ny=ny, lx=lx, ly=ly)

    diagnostics = calculate_diagnostics(
        times,
        rho,
        vx,
        vy,
        grid.dx,
        grid.dy,
    )

    print("\nNumerical checks")
    print("----------------")

    print("Minimum density:", rho.min())
    print("Maximum density:", rho.max())
    print("Maximum speed:", np.sqrt(vx**2 + vy**2).max())

    print(
        "Polarization range:",
        diagnostics["polarization"].min(),
        diagnostics["polarization"].max(),
    )

    # assert np.all(rho >= -1e-10), "Negative density detected"

    # assert np.all(
    #     diagnostics["polarization"] <= 1.0 + 1e-10
    # ), "Polarization exceeds 1"

    print("Simulation summary")
    print("------------------")
    print(f"Number of frames: {len(times)}")
    print(f"Time interval: {times[0]:.3f} - {times[-1]:.3f}")
    print(f"Initial mass: {diagnostics['total_mass'][0]:.6f}")
    print(f"Final mass: {diagnostics['total_mass'][-1]:.6f}")
    print(f"Initial polarization: {diagnostics['polarization'][0]:.6f}")
    print(f"Final polarization: {diagnostics['polarization'][-1]:.6f}")
    print(f"Final mean speed: {diagnostics['mean_speed'][-1]:.6f}")

    plot_snapshots(times, rho, vx, vy, x, y)
    plot_diagnostics(times, diagnostics)
    plot_density_profile(times, rho, x)

    # Existing animation function from visualization.py.
    fig_animation, animation = animate_fields(
        grid,
        times,
        rho,
        vx,
        vy,
        skip=4,
        interval=50,
    )

    # Keep the animation object alive until plt.show().
    plt.show()


if __name__ == "__main__":
    main()
