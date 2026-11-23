"""Nonlinear spatial Toner–Tu evolution with vorticity and convergence plots.

Starts from weak seeded random fluctuations, so vortical structures are not
imposed in the initial condition. This is a numerical experiment, not an
analytical solution or a claim that every parameter set produces vortices.
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from toner_tu.diagnostics import mean_speed, minimum_density, polarization, total_mass
from toner_tu.fields import Fields, Grid
from toner_tu.operators import vorticity
from toner_tu.parameters import TonerTuParameters
from toner_tu.solver import integrate
from toner_tu.visualization import animate_fields_with_vorticity

GRID_N, DOMAIN_LENGTH, FINAL_TIME = 48, 24.0, 1.0
DT_VALUES = (0.004, 0.002, 0.001, 0.0005)
SEED = 42


def initial_state(grid):
    """Seed long-wave random fluctuations around a weakly ordered state."""
    rng = np.random.default_rng(SEED)
    noise_x = rng.normal(size=(grid.ny, grid.nx))
    noise_y = rng.normal(size=(grid.ny, grid.nx))
    kx = 2.0*np.pi*np.fft.fftfreq(grid.nx, d=grid.dx)
    ky = 2.0*np.pi*np.fft.fftfreq(grid.ny, d=grid.dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k2 = kx_grid**2 + ky_grid**2
    cutoff = 3.0*2.0*np.pi/min(grid.lx, grid.ly)
    spectral_filter = np.exp(-(k2/cutoff**2)**2)
    noise_x = np.fft.ifft2(np.fft.fft2(noise_x)*spectral_filter).real
    noise_y = np.fft.ifft2(np.fft.fft2(noise_y)*spectral_filter).real
    noise_x *= 0.08/np.sqrt(np.mean(noise_x**2))
    noise_y *= 0.08/np.sqrt(np.mean(noise_y**2))
    return Fields(
        np.ones((grid.ny, grid.nx)),
        0.05 + noise_x,
        noise_y,
    )


def model_parameters():
    return TonerTuParameters(alpha=1.0, beta=1.0, sigma=0.6,
                             diffusion=0.08, lambda_=2.0)


def run(dt, save_history=False):
    grid = Grid(GRID_N, GRID_N, DOMAIN_LENGTH, DOMAIN_LENGTH)
    fields = initial_state(grid)
    params = model_parameters()
    steps = round(FINAL_TIME / dt)
    dt = FINAL_TIME / steps
    sample_every = max(1, steps // 40)
    history = {"times": [0.0], "rho": [fields.rho.copy()],
               "vx": [fields.vx.copy()], "vy": [fields.vy.copy()]}
    for index in range(steps):
        fields = integrate(fields, grid, params, dt, n_steps=1)
        if save_history and ((index + 1) % sample_every == 0 or index + 1 == steps):
            history["times"].append((index + 1)*dt)
            history["rho"].append(fields.rho.copy())
            history["vx"].append(fields.vx.copy())
            history["vy"].append(fields.vy.copy())
    omega = vorticity(fields.vx, fields.vy, grid)
    metrics = {
        "dt": dt, "steps": steps,
        "mass": total_mass(fields, grid),
        "min_density": minimum_density(fields),
        "polarization": polarization(fields),
        "mean_speed": mean_speed(fields),
        "vorticity_rms": float(np.sqrt(np.mean(omega**2))),
        "enstrophy": float(0.5*np.mean(omega**2)),
    }
    return fields, grid, metrics, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/validation")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    finest, _, finest_metrics, _ = run(DT_VALUES[-1])
    reference_norm = np.sqrt(np.mean(finest.rho**2 + finest.vx**2 + finest.vy**2))
    rows = []
    previous_error = None
    for dt in DT_VALUES:
        fields, _, metrics, _ = run(dt)
        error = np.sqrt(np.mean(
            (fields.rho-finest.rho)**2
            + (fields.vx-finest.vx)**2
            + (fields.vy-finest.vy)**2
        )) / reference_norm
        order = "" if previous_error is None or error == 0 else np.log2(previous_error/error)
        rows.append({**metrics, "relative_error_to_finest_dt": float(error),
                     "observed_order_between_runs": float(order) if order != "" else ""})
        previous_error = error

    with (output/"nonlinear_spatial_dynamics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    sample, grid, _, history = run(DT_VALUES[2], save_history=True)
    omega = vorticity(sample.vx, sample.vy, grid)
    initial_omega = vorticity(history["vx"][0], history["vy"][0], grid)
    initial_omega_rms = float(np.sqrt(np.mean(initial_omega**2)))
    final_omega_rms = float(np.sqrt(np.mean(omega**2)))
    initial_enstrophy = float(0.5*np.mean(initial_omega**2))
    final_enstrophy = float(0.5*np.mean(omega**2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].imshow(np.sqrt(sample.vx**2+sample.vy**2), origin="lower",
                   extent=(0, grid.lx, 0, grid.ly), aspect="equal")
    axes[0].set_title("Local speed")
    image = axes[1].imshow(omega, origin="lower", extent=(0, grid.lx, 0, grid.ly),
                           aspect="equal", cmap="RdBu_r")
    axes[1].set_title("Vorticity")
    fig.colorbar(image, ax=axes[1], label="Vorticity")
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(output/"nonlinear_spatial_dynamics.png", dpi=160)

    with (output/"nonlinear_spatial_global_diagnostics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time", "polarization", "mean_speed", "vorticity_rms",
                         "enstrophy", "mass", "min_density"))
        for time, rho, vx, vy in zip(history["times"], history["rho"],
                                     history["vx"], history["vy"]):
            state = Fields(rho, vx, vy)
            w = vorticity(vx, vy, grid)
            writer.writerow((time, polarization(state), mean_speed(state),
                             np.sqrt(np.mean(w**2)), 0.5*np.mean(w**2),
                             total_mass(state, grid), minimum_density(state)))

    if not args.no_show:
        animate_fields_with_vorticity(
            grid, history["times"], history["rho"], history["vx"],
            history["vy"], skip=3, interval=90,
        )
        plt.show()

    print("dt, relative L2 error to finest, polarization, vorticity RMS, enstrophy")
    for row in rows:
        print(f'{row["dt"]:.6g}, {row["relative_error_to_finest_dt"]:.6e}, '
              f'{row["polarization"]:.6e}, {row["vorticity_rms"]:.6e}, '
              f'{row["enstrophy"]:.6e}')
    print(f"Initial vorticity RMS/enstrophy: {initial_omega_rms:.6e}, {initial_enstrophy:.6e}")
    print(f"Final vorticity RMS/enstrophy:   {final_omega_rms:.6e}, {final_enstrophy:.6e}")
    print(f"Vorticity RMS amplification:     {final_omega_rms/initial_omega_rms:.6e}")
    print(f"Saved nonlinear spatial diagnostics in {output}")


if __name__ == "__main__":
    main()
