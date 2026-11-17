
import numpy as np
import pytest

from toner_tu.fields import Fields, Grid
from toner_tu.initial_conditions import (
    uniform_state,
    random_velocity_state,
)
from toner_tu.diagnostics import (
    total_mass,
    mean_density,
    minimum_density,
    mean_velocity,
    mean_speed,
    polarization,
    kinetic_energy_density,
)


@pytest.fixture
def grid():
    return Grid(nx=10, ny=10, lx=10.0, ly=10.0)


def test_uniform_initial_state(grid):
    fields = uniform_state(grid, rho0=2.0, vx0=1.0, vy0=0.0)

    assert np.all(fields.rho == 2.0)
    assert np.all(fields.vx == 1.0)
    assert np.all(fields.vy == 0.0)


def test_random_initial_state_is_reproducible(grid):
    a = random_velocity_state(grid, seed=42)
    b = random_velocity_state(grid, seed=42)

    assert np.array_equal(a.vx, b.vx)
    assert np.array_equal(a.vy, b.vy)


def test_mass(grid):
    fields = uniform_state(grid, rho0=2.0)

    assert total_mass(fields, grid) == pytest.approx(200.0)


def test_density_diagnostics(grid):
    fields = uniform_state(grid, rho0=2.0)

    assert mean_density(fields) == pytest.approx(2.0)
    assert minimum_density(fields) == pytest.approx(2.0)


def test_mean_velocity(grid):
    fields = uniform_state(grid, vx0=3.0, vy0=4.0)

    assert mean_velocity(fields) == pytest.approx((3.0, 4.0))
    assert mean_speed(fields) == pytest.approx(5.0)


def test_polarization(grid):
    fields = uniform_state(grid, vx0=3.0, vy0=4.0)

    assert polarization(fields) == pytest.approx(1.0)


def test_kinetic_energy(grid):
    fields = uniform_state(grid, rho0=2.0, vx0=3.0, vy0=4.0)

    assert kinetic_energy_density(fields) == pytest.approx(25.0)
