import numpy as np
import pytest

from toner_tu.fields import Fields
from toner_tu.boundary_conditions import (
    ChannelBoundaryConditions,
    apply_channel_boundary_conditions,
)


@pytest.fixture
def fields():
    shape = (8, 12)

    return Fields(
        rho=np.full(shape, 2.0),
        vx=np.full(shape, 0.2),
        vy=np.full(shape, 0.1),
    )


def test_inlet_conditions(fields):
    boundary = ChannelBoundaryConditions(
        rho_in=1.5,
        vx_in=0.5,
        vy_in=0.0,
    )

    result = apply_channel_boundary_conditions(fields, boundary)

    assert np.all(result.rho[1:-1, 0] == 1.5)
    assert np.all(result.vx[1:-1, 0] == 0.5)
    assert np.all(result.vy[1:-1, 0] == 0.0)


def test_outlet_zero_gradient(fields):
    boundary = ChannelBoundaryConditions()

    result = apply_channel_boundary_conditions(fields, boundary)

    assert np.allclose(result.rho[:, -1], result.rho[:, -2])
    assert np.allclose(result.vx[:, -1], result.vx[:, -2])
    assert np.allclose(result.vy[:, -1], result.vy[:, -2])


def test_impermeable_walls(fields):
    boundary = ChannelBoundaryConditions()

    result = apply_channel_boundary_conditions(fields, boundary)

    assert np.all(result.vy[0, :] == 0.0)
    assert np.all(result.vy[-1, :] == 0.0)

    assert np.allclose(result.vx[0, :], result.vx[1, :])
    assert np.allclose(result.vx[-1, :], result.vx[-2, :])


def test_input_fields_are_not_modified(fields):
    original_rho = fields.rho.copy()
    original_vx = fields.vx.copy()
    original_vy = fields.vy.copy()

    apply_channel_boundary_conditions(
        fields,
        ChannelBoundaryConditions(),
    )

    assert np.array_equal(fields.rho, original_rho)
    assert np.array_equal(fields.vx, original_vx)
    assert np.array_equal(fields.vy, original_vy)

