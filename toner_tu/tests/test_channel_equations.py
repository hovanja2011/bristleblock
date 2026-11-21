
import numpy as np
import pytest

from toner_tu.channel_fluxes import compute_density_rhs
from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.boundary_conditions import ChannelBoundaryConditions
from toner_tu.channel_equations import compute_channel_rhs


@pytest.fixture
def grid():
    return Grid(nx=16, ny=16, lx=10.0, ly=10.0)


@pytest.fixture
def params():
    return TonerTuParameters(
        alpha=1.0,
        beta=1.0,
        sigma=1.0,
        diffusion=0.1,
        lambda_=1.0,
    )


@pytest.fixture
def boundary():
    return ChannelBoundaryConditions(
        rho_in=1.0,
        vx_in=1.0,
        vy_in=0.0,
    )


def test_rhs_shape(grid, params, boundary):
    shape = (grid.ny, grid.nx)

    fields = Fields(
        rho=np.ones(shape),
        vx=np.full(shape, 0.5),
        vy=np.zeros(shape),
    )

    rhs = compute_channel_rhs(fields, grid, params, boundary)

    assert rhs.rho.shape == shape
    assert rhs.vx.shape == shape
    assert rhs.vy.shape == shape


def test_rhs_is_finite(grid, params, boundary):
    rng = np.random.default_rng(42)
    shape = (grid.ny, grid.nx)

    fields = Fields(
        rho=1.0 + 0.01 * rng.random(shape),
        vx=0.5 + 0.01 * rng.normal(size=shape),
        vy=0.01 * rng.normal(size=shape),
    )

    rhs = compute_channel_rhs(fields, grid, params, boundary)

    assert np.all(np.isfinite(rhs.rho))
    assert np.all(np.isfinite(rhs.vx))
    assert np.all(np.isfinite(rhs.vy))


def test_uniform_inlet_flow(grid, params, boundary):
    shape = (grid.ny, grid.nx)

    fields = Fields(
        rho=np.ones(shape),
        vx=np.full(shape, 1.0),
        vy=np.zeros(shape),
    )

    rhs = compute_channel_rhs(fields, grid, params, boundary)

    # The uniform ordered velocity is an equilibrium
    # in the interior of the channel.
    assert np.allclose(rhs.rho[2:-2, 2:-2], 0.0)
    assert np.allclose(rhs.vx[2:-2, 2:-2], 0.0)
    assert np.allclose(rhs.vy[2:-2, 2:-2], 0.0)


def test_input_fields_are_not_modified(grid, params, boundary):
    shape = (grid.ny, grid.nx)

    fields = Fields(
        rho=np.ones(shape),
        vx=np.full(shape, 0.2),
        vy=np.zeros(shape),
    )

    original_rho = fields.rho.copy()
    original_vx = fields.vx.copy()
    original_vy = fields.vy.copy()

    compute_channel_rhs(fields, grid, params, boundary)

    assert np.array_equal(fields.rho, original_rho)
    assert np.array_equal(fields.vx, original_vx)
    assert np.array_equal(fields.vy, original_vy)


@pytest.mark.parametrize(
    "vx_value, inlet_velocity",
    [
        (0.4, 0.4),    # Forward flow
        (-0.4, -0.4),  # Reverse flow
    ],
)
def test_production_rhs_uses_canonical_density_fluxes(
    grid, params, vx_value, inlet_velocity
):
    rng = np.random.default_rng(42)
    shape = (grid.ny, grid.nx)

    fields = Fields(
        rho=rng.uniform(0.5, 1.5, shape),
        vx=np.full(shape, vx_value),
        vy=rng.uniform(-0.1, 0.1, shape),
    )

    boundary = ChannelBoundaryConditions(
        rho_in=1.4,
        vx_in=inlet_velocity,
        vy_in=0.0,
    )

    expected = compute_density_rhs(
        fields.rho,
        fields.vx,
        fields.vy,
        grid,
        boundary,
    )

    production_rhs = compute_channel_rhs(
        fields,
        grid,
        params,
        boundary,
    )

    assert np.allclose(
        production_rhs.rho,
        expected,
        rtol=1e-13,
        atol=1e-13,
    )