
from pathlib import Path

import numpy as np

from toner_tu.fields import Grid, Fields
from toner_tu.parameters import TonerTuParameters
from toner_tu.boundary_conditions import (
    ChannelBoundaryConditions,
    apply_channel_boundary_conditions,
)
from toner_tu.channel_solver import step


def finite_swarm_initial_condition(
    grid: Grid,
    rho0: float = 1.0,
    initial_length: float = 5.0,
    speed: float = 1.0,
    angle_spread: float = np.pi,
    seed: int = 42,
) -> Fields:
    """
    Finite swarm initially localized near the left side of the channel.

    Heading angles are uniformly distributed in
    [-angle_spread / 2, angle_spread / 2].
    """

    if rho0 < 0:
        raise ValueError("rho0 must be non-negative")

    if not 0 < initial_length <= grid.lx:
        raise ValueError("initial_length must be in (0, grid.lx]")

    if speed < 0:
        raise ValueError("speed must be non-negative")

    if not 0 <= angle_spread <= 2 * np.pi:
        raise ValueError("angle_spread must be in [0, 2*pi]")

    rng = np.random.default_rng(seed)

    x = np.arange(grid.nx) * grid.dx

    # The initial swarm occupies only the left part of the channel.
    inside = (x > 0) & (x <= initial_length)

    rho = np.zeros((grid.ny, grid.nx))
    vx = np.zeros_like(rho)
    vy = np.zeros_like(rho)

    angles = rng.uniform(
        -angle_spread / 2,
        angle_spread / 2,
        size=(grid.ny, grid.nx),
    )

    rho[:, inside] = rho0
    vx[:, inside] = speed * np.cos(angles[:, inside])
    vy[:, inside] = speed * np.sin(angles[:, inside])

    return Fields(rho=rho, vx=vx, vy=vy)


def calculate_diagnostics(fields: Fields, grid: Grid) -> dict:
    """Calculate diagnostics weighted by the density."""

    rho = fields.rho
    speed = np.sqrt(fields.vx**2 + fields.vy**2)

    total_mass = np.sum(rho) * grid.dx * grid.dy

    if total_mass > 1e-12:
        mean_speed = np.sum(rho * speed) / np.sum(rho)

        mean_vx = np.sum(rho * fields.vx) / np.sum(rho)
        mean_vy = np.sum(rho * fields.vy) / np.sum(rho)

        mean_velocity = np.sqrt(mean_vx**2 + mean_vy**2)

        mean_local_speed = np.sum(rho * speed) / np.sum(rho)

        if mean_local_speed > 1e-12:
            polarization = mean_velocity / mean_local_speed
        else:
            polarization = 0.0
    else:
        mean_speed = 0.0
        mean_velocity = 0.0
        polarization = 0.0

    return {
        "mass": total_mass,
        "mean_speed": mean_speed,
        "mean_velocity": mean_velocity,
        "polarization": polarization,
        "min_density": np.min(rho),
        "max_density": np.max(rho),
    }


def run_experiment():
    # --------------------------------------------------
    # 1. Geometry
    # --------------------------------------------------

    grid = Grid(
        nx=128,
        ny=48,
        lx=40.0,
        ly=12.0,
    )

    x = np.arange(grid.nx) * grid.dx
    y = np.arange(grid.ny) * grid.dy

    # --------------------------------------------------
    # 2. Toner-Tu parameters
    # --------------------------------------------------

    params = TonerTuParameters(
        alpha=1.0,
        beta=1.0,
        sigma=1.0,
        diffusion=0.5,
        lambda_=1.0,
    )

    # Preferred speed in a homogeneous ordered state:
    # v0 = sqrt(alpha / beta) = 1.

    # --------------------------------------------------
    # 3. Boundary conditions
    # --------------------------------------------------

    boundary = ChannelBoundaryConditions(
        rho_in=0.0,
        vx_in=0.0,
        vy_in=0.0,
    )

    # --------------------------------------------------
    # 4. Initial finite swarm
    # --------------------------------------------------

    fields = finite_swarm_initial_condition(
        grid=grid,
        rho0=1.0,
        initial_length=5.0,
        speed=1.0,
        angle_spread=np.pi,
        seed=42,
    )

    fields = apply_channel_boundary_conditions(fields, boundary)

    # --------------------------------------------------
    # 5. Time integration
    # --------------------------------------------------

    dt = 0.001
    n_steps = 20000
    sample_every = 100

    times = []

    rho_history = []
    vx_history = []
    vy_history = []

    mass_history = []
    mean_speed_history = []
    mean_velocity_history = []
    polarization_history = []

    print("Starting finite-swarm experiment")
    print(f"Grid: {grid.nx} x {grid.ny}")
    print(f"Domain: {grid.lx} x {grid.ly}")
    print(f"Time step: {dt}")
    print(f"Number of steps: {n_steps}")
    print(f"Initial swarm length: 5.0")
    print(f"Initial angular spread: 180 degrees")
    print()

    # --------------------------------------------------
    # 6. Main loop
    # --------------------------------------------------

    for i in range(n_steps + 1):

        if i % sample_every == 0:
            diagnostics = calculate_diagnostics(fields, grid)

            times.append(i * dt)

            rho_history.append(fields.rho.copy())
            vx_history.append(fields.vx.copy())
            vy_history.append(fields.vy.copy())

            mass_history.append(diagnostics["mass"])
            mean_speed_history.append(diagnostics["mean_speed"])
            mean_velocity_history.append(diagnostics["mean_velocity"])
            polarization_history.append(diagnostics["polarization"])

            print(
                f"Step {i:5d}/{n_steps}: "
                f"mass = {diagnostics['mass']:.4f}, "
                f"mean speed = {diagnostics['mean_speed']:.4f}, "
                f"polarization = {diagnostics['polarization']:.4f}, "
                f"max rho = {diagnostics['max_density']:.4f}"
            )

        if i < n_steps:
            fields = step(
                fields=fields,
                grid=grid,
                params=params,
                boundary=boundary,
                dt=dt,
            )

    # --------------------------------------------------
    # 7. Save results
    # --------------------------------------------------

    output_path = Path(__file__).resolve().parent / "channel_history.npz"

    np.savez_compressed(
        output_path,
        times=np.asarray(times),
        rho=np.asarray(rho_history),
        vx=np.asarray(vx_history),
        vy=np.asarray(vy_history),
        x=x,
        y=y,
        mass=np.asarray(mass_history),
        mean_speed=np.asarray(mean_speed_history),
        mean_velocity=np.asarray(mean_velocity_history),
        polarization=np.asarray(polarization_history),
        dt=dt,
        n_steps=n_steps,
    )

    print()
    print(f"Experiment finished.")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    run_experiment()
