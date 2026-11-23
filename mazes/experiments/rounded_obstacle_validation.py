"""Grid-refinement and no-flux diagnostics for rounded internal maze walls."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mazes import CircleObstacle, MazeGeometry, RoundedPolyline
from toner_tu.initial_conditions import uniform_state
from toner_tu.parameters import TonerTuParameters
from toner_tu.solver import integrate

WIDTH, HEIGHT = 4.0, 3.0
CIRCLE_RADIUS = 0.30
WALL_POINTS = ((0.8, 0.9), (2.3, 0.9), (2.3, 1.8))
WALL_RADIUS = 0.12


def make_geometry():
    return MazeGeometry(WIDTH, HEIGHT, obstacles=(
        RoundedPolyline(WALL_POINTS, WALL_RADIUS),
        CircleObstacle((3.1, 2.2), CIRCLE_RADIUS),
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/validation")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    geometry = make_geometry()

    wall_length = sum(
        np.linalg.norm(np.asarray(b)-np.asarray(a))
        for a, b in zip(WALL_POINTS[:-1], WALL_POINTS[1:])
    )
    # Area of a non-self-intersecting open polyline thickened by a disk,
    # plus the disjoint circular obstacle.
    exact_area = (
        2.0*wall_length*WALL_RADIUS + np.pi*WALL_RADIUS**2
        + np.pi*CIRCLE_RADIUS**2
    )
    rows = []
    finest_grid = finest_mask = None
    for nx in (40, 80, 160):
        ny = round(nx*HEIGHT/WIDTH)
        grid = geometry.grid(nx, ny)
        operators = geometry.operators(grid)
        mask = operators.solid_mask
        area = np.count_nonzero(mask)*grid.dx*grid.dy
        state = uniform_state(grid, rho0=1.0, vx0=0.3, vy0=0.1)
        state.rho += 0.03*np.cos(2*np.pi*grid.x/WIDTH)*np.cos(2*np.pi*grid.y/HEIGHT)
        state = operators.mask_fields(state)
        mass0 = float(np.sum(state.rho)*grid.dx*grid.dy)
        params = TonerTuParameters(
            alpha=0.0, beta=1.0, sigma=0.0,
            diffusion=0.0, lambda_=0.0,
        )
        evolved = integrate(
            state, grid, params, dt=1e-4, n_steps=50, domain=operators
        )
        mass1 = float(np.sum(evolved.rho)*grid.dx*grid.dy)
        rows.append({
            "nx": nx, "ny": ny, "solid_area": area,
            "exact_solid_area": exact_area,
            "absolute_area_error": abs(area-exact_area),
            "relative_mass_drift": abs(mass1-mass0)/mass0,
            "minimum_fluid_density": float(np.min(evolved.rho[~mask])),
        })
        finest_grid, finest_mask = grid, mask

    with (output/"rounded_obstacle_validation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].imshow(
        finest_mask, origin="lower", extent=(0, WIDTH, 0, HEIGHT),
        interpolation="nearest", aspect="equal",
    )
    axes[0].set_title("Rounded polyline wall and circular obstacle")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[1].loglog(
        [r["nx"] for r in rows],
        [r["absolute_area_error"] for r in rows], "o-",
    )
    axes[1].set_xlabel("Grid nx")
    axes[1].set_ylabel("Absolute solid-area error")
    axes[1].set_title("Geometry grid refinement")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output/"rounded_obstacle_validation.png", dpi=160)

    print("nx, area error, relative mass drift, minimum fluid density")
    for row in rows:
        print(
            f'{row["nx"]}, {row["absolute_area_error"]:.6e}, '
            f'{row["relative_mass_drift"]:.6e}, '
            f'{row["minimum_fluid_density"]:.6e}'
        )
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
