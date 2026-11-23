"""Periodic linear-acoustic benchmark for the Toner–Tu solver.

Run from the repository root:
    python toner_tu/experiments/periodic_acoustic_convergence.py

The continuum reference is a small-amplitude 1D acoustic Fourier mode on a
periodic domain. The output reports normalized L2 errors and observed orders.
"""
import numpy as np

from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.solver import integrate


AMPLITUDE = 1.0e-5
FINAL_TIME = 0.1


def run(nx):
    grid = Grid(nx=nx, ny=4, lx=1.0, ly=1.0)
    x, _ = grid.coordinates()
    rho = 1.0 + AMPLITUDE * np.cos(2.0 * np.pi * x)
    fields = Fields(rho, np.zeros_like(rho), np.zeros_like(rho))
    params = TonerTuParameters(
        alpha=0.0,
        beta=1.0,
        sigma=1.0,
        diffusion=0.0,
        lambda_=0.0,
    )
    dt = FINAL_TIME / (nx * nx)
    result = integrate(fields, grid, params, dt=dt, n_steps=nx * nx)

    phase = 2.0 * np.pi * FINAL_TIME
    rho_exact = 1.0 + AMPLITUDE * np.cos(2.0 * np.pi * x) * np.cos(phase)
    vx_exact = AMPLITUDE * np.sin(2.0 * np.pi * x) * np.sin(phase)
    return float(np.sqrt(np.mean(
        ((result.rho - rho_exact) / AMPLITUDE) ** 2
        + ((result.vx - vx_exact) / AMPLITUDE) ** 2
    )))


def main():
    resolutions = (16, 32, 64)
    errors = [run(nx) for nx in resolutions]
    print("nx    normalized L2 error    observed order")
    previous = None
    for nx, error in zip(resolutions, errors):
        order = float("nan") if previous is None else np.log2(previous / error)
        print(f"{nx:<5d} {error:<22.8e} {order:.4f}")
        previous = error


if __name__ == "__main__":
    main()
