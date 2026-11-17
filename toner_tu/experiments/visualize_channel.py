
import numpy as np
import matplotlib.pyplot as plt

from toner_tu.fields import Grid
from toner_tu.visualization import (
    animate_fields_with_vorticity,
    plot_diagnostics,
)


def main():
    # Load saved simulation data
    data = np.load("toner_tu/experiments/channel_history.npz")

    times = data["times"]
    rho = data["rho"]
    vx = data["vx"]
    vy = data["vy"]

    # Reconstruct the original grid
    ny, nx = rho.shape[1:]

    grid = Grid(
        nx=nx,
        ny=ny,
        lx=40.0,
        ly=12.0,
    )

    print("Data loaded:")
    print("Frames:", len(times))
    print("Grid:", nx, "x", ny)

    # --------------------------------
    # Diagnostics
    # --------------------------------

    speeds = np.mean(
        np.sqrt(vx**2 + vy**2),
        axis=(1, 2),
    )

    mean_vx = np.mean(vx, axis=(1, 2))
    mean_vy = np.mean(vy, axis=(1, 2))

    mean_speed = np.maximum(speeds, 1e-12)

    polarizations = (
        np.sqrt(mean_vx**2 + mean_vy**2) / mean_speed
    )

    min_densities = np.min(
        rho,
        axis=(1, 2),
    )

    fig_diagnostics, axes = plot_diagnostics(
        times,
        polarizations,
        speeds,
        min_densities,
    )

    # --------------------------------
    # Animation
    # --------------------------------

    fig_animation, animation = animate_fields_with_vorticity(
        grid=grid,
        times=times,
        rho_history=rho,
        vx_history=vx,
        vy_history=vy,
        skip=4,
        interval=50,
    )

    print("Animation created.")
    print("Close the animation window to finish.")

    # Keep animation object alive while displaying it
    plt.show()


if __name__ == "__main__":
    main()
