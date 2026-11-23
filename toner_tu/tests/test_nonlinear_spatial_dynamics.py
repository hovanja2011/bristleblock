"""Smoke and invariant tests for nonlinear spatial Toner–Tu dynamics."""
import numpy as np
import pytest

from toner_tu.diagnostics import total_mass
from toner_tu.fields import Grid
from toner_tu.initial_conditions import random_velocity_state
from toner_tu.operators import vorticity
from toner_tu.parameters import TonerTuParameters
from toner_tu.solver import integrate


def test_nonlinear_spatial_run_remains_finite_and_mass_conservative():
    grid = Grid(24, 24, 12.0, 12.0)
    state = random_velocity_state(grid, velocity_scale=0.2, seed=7)
    params = TonerTuParameters(
        alpha=1.0, beta=1.0, sigma=0.6,
        diffusion=0.08, lambda_=2.0,
    )
    initial_mass = total_mass(state, grid)
    result = integrate(state, grid, params, dt=0.001, n_steps=20)
    assert np.isfinite(result.rho).all()
    assert np.isfinite(result.vx).all()
    assert np.isfinite(result.vy).all()
    assert result.rho.min() > 0.0
    assert total_mass(result, grid) == pytest.approx(initial_mass, abs=1e-10)
    assert np.sqrt(np.mean(vorticity(result.vx, result.vy, grid)**2)) > 0.0
