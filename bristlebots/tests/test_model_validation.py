import numpy as np
import pytest

from bristlebots.contacts import ContactParameters, ContactState, robot_wall_contact
from bristlebots.geometry import Wall
from bristlebots.robot import Robot
from bristlebots.simulation import step, step_robots


@pytest.mark.parametrize("integrator", [step, step_robots])
def test_central_propulsion_does_not_generate_torque(integrator):
    robot = Robot(
        position=(0.0, 0.0),
        velocity=(0.0, 0.0),
        theta=0.4,
        radius=0.5,
        mass=2.0,
        propulsion_force=3.0,
    )

    if integrator is step:
        diagnostics = integrator(robot, 1e-3)
    else:
        diagnostics = integrator([robot], 1e-3)[0]

    assert robot.omega == pytest.approx(0.0, abs=1e-15)
    assert diagnostics["active_torque"] == 0.0
    assert diagnostics["total_torque"] == pytest.approx(0.0, abs=1e-15)
    assert "propulsion_torque" not in diagnostics


@pytest.mark.parametrize(
    "kwargs",
    [
        {"radius": 0.0},
        {"mass": -1.0},
        {"translational_drag": -1.0},
        {"rotational_drag": np.nan},
        {"theta": np.inf},
        {"position": (0.0, np.inf)},
        {"velocity": (0.0,)},
    ],
)
def test_robot_rejects_invalid_physical_parameters(kwargs):
    arguments = {
        "position": (0.0, 0.0),
        "velocity": (0.0, 0.0),
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        Robot(**arguments)


@pytest.mark.parametrize(
    "values",
    [
        (np.nan, 0.0, 0.0, 0.0, 1.0, 0.0),
        (1.0, np.inf, 0.0, 0.0, 1.0, 0.0),
    ],
)
def test_contact_parameters_must_be_finite(values):
    with pytest.raises(ValueError, match="finite"):
        ContactParameters(*values)


def test_wall_rejects_non_finite_or_malformed_endpoints():
    with pytest.raises(ValueError):
        Wall((0.0, 0.0), (np.inf, 1.0))
    with pytest.raises(ValueError):
        Wall((0.0,), (1.0, 1.0))


def test_wall_contact_history_is_rotated_with_contact_tangent():
    wall = Wall((0.0, 0.0), (1.0, 0.0))
    robot = Robot(
        position=(1.9, 0.0),
        velocity=(0.0, 0.0),
        radius=1.2,
        mass=1.0,
    )
    parameters = ContactParameters(
        normal_stiffness=100.0,
        normal_damping=0.0,
        mu_static=100.0,
        mu_kinetic=100.0,
        tangential_stiffness=10.0,
    )
    state = ContactState(
        tangential_displacement=0.1,
        previous_tangent=np.array([0.0, 1.0]),
    )

    angle = np.pi / 3
    robot.position[:] = (1.0 + 0.9 * np.cos(angle), 0.9 * np.sin(angle))
    robot_wall_contact(robot, wall, 1e-6, state, parameters)

    assert state.tangential_displacement == pytest.approx(0.1 * np.cos(angle))
    np.testing.assert_allclose(
        state.previous_tangent,
        (-np.sin(angle), np.cos(angle)),
        atol=1e-14,
    )


def test_free_speed_is_defined_when_linear_drag_is_zero():
    driven = Robot((0.0, 0.0), (0.0, 0.0), propulsion_force=1.0)
    unpowered = Robot((0.0, 0.0), (0.0, 0.0))
    assert np.isinf(driven.free_speed)
    assert unpowered.free_speed == 0.0


def test_contact_parameters_are_immutable_after_validation():
    parameters = ContactParameters(
        normal_stiffness=100.0,
        normal_damping=0.0,
        mu_static=0.5,
        mu_kinetic=0.3,
        tangential_stiffness=10.0,
    )
    with pytest.raises(AttributeError):
        parameters.normal_stiffness = -1.0
