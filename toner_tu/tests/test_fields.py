import numpy as np
import pytest

from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters


def test_grid_spacing():
    grid = Grid(nx=100, ny=50, lx=10.0, ly=5.0)
    assert grid.dx == pytest.approx(0.1)
    assert grid.dy == pytest.approx(0.1)


def test_grid_coordinates():
    grid = Grid(nx=10, ny=5, lx=10.0, ly=5.0)
    x, y = grid.coordinates()
    assert x.shape == (5, 10)
    assert y.shape == (5, 10)


def test_grid_rejects_invalid_dimensions_and_lengths():
    for args in [(1, 5, 1.0, 1.0), (5, 0, 1.0, 1.0),
                 (5, 5, 0.0, 1.0), (5, 5, 1.0, np.inf),
                 (5.5, 5, 1.0, 1.0)]:
        with pytest.raises(ValueError):
            Grid(*args)


def test_fields_copy():
    rho = np.ones((5, 10))
    vx = np.zeros((5, 10))
    vy = np.zeros((5, 10))
    fields = Fields(rho, vx, vy)
    copied = fields.copy()
    copied.rho[0, 0] = 2.0
    assert fields.rho[0, 0] == 1.0


def test_fields_reject_incompatible_arrays():
    with pytest.raises(ValueError):
        Fields(np.ones((3, 4)), np.ones((3, 4)), np.ones((4, 3)))


def test_ordered_speed():
    params = TonerTuParameters(alpha=1.0, beta=2.0, sigma=1.0, diffusion=0.1)
    assert params.ordered_speed == pytest.approx(2**-0.5)


def test_invalid_beta():
    with pytest.raises(ValueError):
        TonerTuParameters(alpha=1.0, beta=0.0, sigma=1.0, diffusion=0.1)


def test_non_finite_parameters_are_rejected():
    with pytest.raises(ValueError):
        TonerTuParameters(alpha=np.nan, beta=1.0, sigma=1.0, diffusion=0.1)


def test_initial_condition_builders_reject_non_finite_values():
    from toner_tu.initial_conditions import uniform_state, random_velocity_state

    grid = Grid(nx=4, ny=4, lx=1.0, ly=1.0)
    with pytest.raises(ValueError):
        uniform_state(grid, rho0=np.nan)
    with pytest.raises(ValueError):
        uniform_state(grid, vx0=np.inf)
    with pytest.raises(ValueError):
        random_velocity_state(grid, velocity_scale=np.nan)
