
import numpy as np
import matplotlib.pyplot as plt

from toner_tu.fields import Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.initial_conditions import random_velocity_state
from toner_tu.solver import integrate

from toner_tu.diagnostics import (
    polarization,
    mean_speed,
    minimum_density,
    total_mass,
)

from toner_tu.visualization import (
    plot_diagnostics,
    animate_fields_with_vorticity,
)

# --------------------------------------------------
# 1. Grid
# --------------------------------------------------

grid = Grid(
    nx=64,
    ny=64,
    lx=40.0,
    ly=40.0,
)


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


# --------------------------------------------------
# 3. Initial conditions
# --------------------------------------------------

fields = random_velocity_state(
    grid,
    rho0=1.0,
    velocity_scale=0.1,
    seed=42,
)


# --------------------------------------------------
# 4. Simulation parameters
# --------------------------------------------------

dt = 0.001
n_steps = 5000
sample_every = 100

n_frames = n_steps // sample_every + 1

times = np.zeros(n_frames)

polarizations = np.zeros(n_frames)
speeds = np.zeros(n_frames)
min_densities = np.zeros(n_frames)

rho_history = np.zeros(
    (n_frames, grid.ny, grid.nx)
)

vx_history = np.zeros_like(rho_history)
vy_history = np.zeros_like(rho_history)


# --------------------------------------------------
# 5. Initial mass
# --------------------------------------------------

initial_mass = total_mass(fields, grid)


# --------------------------------------------------
# 6. Time integration and data collection
# --------------------------------------------------

frame = 0

for i in range(n_steps + 1):

    if i % sample_every == 0:

        times[frame] = i * dt

        polarizations[frame] = polarization(fields)
        speeds[frame] = mean_speed(fields)
        min_densities[frame] = minimum_density(fields)

        rho_history[frame] = fields.rho
        vx_history[frame] = fields.vx
        vy_history[frame] = fields.vy

        frame += 1

    if i < n_steps:
        fields = integrate(
            fields,
            grid,
            params,
            dt,
            n_steps=1,
        )


# --------------------------------------------------
# 7. Numerical diagnostics
# --------------------------------------------------

final_mass = total_mass(fields, grid)

print("\nSimulation finished")
print("-------------------")

print(f"Initial mass: {initial_mass:.10f}")
print(f"Final mass:   {final_mass:.10f}")
print(f"Mass change:  {final_mass - initial_mass:.3e}")

print()

print(f"Initial polarization: {polarizations[0]:.5f}")
print(f"Final polarization:   {polarizations[-1]:.5f}")

print(f"Initial mean speed: {speeds[0]:.5f}")
print(f"Final mean speed:   {speeds[-1]:.5f}")

print(f"Minimum density: {np.min(rho_history):.6f}")
print(f"Maximum density: {np.max(rho_history):.6f}")

velocity_magnitude = np.sqrt(
    vx_history**2 + vy_history**2
)

print(f"Maximum speed: {np.max(velocity_magnitude):.6f}")


# --------------------------------------------------
# 8. Plot global diagnostics
# --------------------------------------------------

plot_diagnostics(
    times,
    polarizations,
    speeds,
    min_densities,
)


# --------------------------------------------------
# 9. Animate density and velocity
# --------------------------------------------------

fig, animation = animate_fields_with_vorticity(
    grid=grid,
    times=times,
    rho_history=rho_history,
    vx_history=vx_history,
    vy_history=vy_history,
    skip=4,
    interval=100,
)

plt.show()


# --------------------------------------------------
# 10. Show figures
# --------------------------------------------------

plt.show()
