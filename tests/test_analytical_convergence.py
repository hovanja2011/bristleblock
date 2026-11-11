
import numpy as np

from bristlebots.robot import Robot
from bristlebots.geometry import Wall
from bristlebots.contacts import ContactParameters
from bristlebots.simulation import step


# --------------------------------------------------
# Physical parameters
# --------------------------------------------------

MASS = 1.0
RADIUS = 0.5

INITIAL_VX = 20.0
NORMAL_STIFFNESS = 4e5

WALL_POSITION = 1.0


def make_setup():
    """Create a robot initially touching a vertical wall."""

    robot = Robot(
        position=[WALL_POSITION - RADIUS, 0.0],
        velocity=[INITIAL_VX, 0.0],
        radius=RADIUS,
        mass=MASS,
        propulsion_force=0.0,
        translational_drag=0.0,
        rotational_drag=0.0,
    )

    wall = Wall(
        start=[WALL_POSITION, -1.0],
        end=[WALL_POSITION, 5.0],
    )

    parameters = ContactParameters(
        normal_stiffness=NORMAL_STIFFNESS,
        normal_damping=0.0,
        mu_static=0.0,
        mu_kinetic=0.0,
        tangential_stiffness=1e4,
        tangential_damping=0.0,
    )

    return robot, wall, parameters


# --------------------------------------------------
# Analytical solution
# --------------------------------------------------

def analytical_solution(time):
    """
    Exact solution for an undamped linear spring collision.

    delta(t) = v0 / omega * sin(omega * t)
    vx(t)    = v0 * cos(omega * t)
    """

    omega = np.sqrt(NORMAL_STIFFNESS / MASS)

    delta = (
        INITIAL_VX / omega
        * np.sin(omega * time)
    )

    vx = INITIAL_VX * np.cos(omega * time)

    return delta, vx


# --------------------------------------------------
# Numerical solution
# --------------------------------------------------

def numerical_solution(dt, total_time):
    """Integrate the collision using simulation.step()."""

    robot, wall, parameters = make_setup()

    steps = round(total_time / dt)

    time = np.arange(steps + 1) * dt

    delta = [0.0]
    vx = [robot.velocity[0]]

    for _ in range(steps):

        step(
            robot,
            dt,
            walls=[wall],
            contact_parameters=parameters,
        )

        penetration = max(
            robot.position[0] + RADIUS - WALL_POSITION,
            0.0,
        )

        delta.append(penetration)
        vx.append(robot.velocity[0])

    return (
        time,
        np.array(delta),
        np.array(vx),
    )


# --------------------------------------------------
# Error calculation
# --------------------------------------------------

def calculate_errors(dt):
    """
    Compare numerical and analytical solutions
    over the same time interval.
    """

    omega = np.sqrt(NORMAL_STIFFNESS / MASS)

    contact_time = np.pi / omega

    # Stop before contact separation.
    total_time = 0.9 * contact_time

    time, delta_num, vx_num = numerical_solution(
        dt,
        total_time,
    )

    delta_exact, vx_exact = analytical_solution(time)

    error_delta = np.max(
        np.abs(delta_num - delta_exact)
    )

    error_vx = np.max(
        np.abs(vx_num - vx_exact)
    )

    return error_delta, error_vx


# --------------------------------------------------
# Test 1: Numerical solution converges to analytical
# --------------------------------------------------

def test_normal_collision_convergence():

    omega = np.sqrt(NORMAL_STIFFNESS / MASS)
    contact_time = np.pi / omega

    time_steps = contact_time / np.array([
        100,
        200,
        400,
        800,
    ])

    errors_delta = []
    errors_vx = []

    for dt in time_steps:

        error_delta, error_vx = calculate_errors(dt)

        errors_delta.append(error_delta)
        errors_vx.append(error_vx)

    errors_delta = np.array(errors_delta)
    errors_vx = np.array(errors_vx)

    # Errors must decrease as the time step decreases.
    assert np.all(np.diff(errors_delta) < 0.0)
    assert np.all(np.diff(errors_vx) < 0.0)


# --------------------------------------------------
# Test 2: Observed order of convergence
# --------------------------------------------------

def test_observed_order_of_convergence():

    omega = np.sqrt(NORMAL_STIFFNESS / MASS)
    contact_time = np.pi / omega

    time_steps = contact_time / np.array([
        100,
        200,
        400,
        800,
    ])

    errors_delta = []
    errors_vx = []

    for dt in time_steps:

        error_delta, error_vx = calculate_errors(dt)

        errors_delta.append(error_delta)
        errors_vx.append(error_vx)

    errors_delta = np.array(errors_delta)
    errors_vx = np.array(errors_vx)

    # Calculate observed convergence order.
    order_delta = (
        np.log(errors_delta[:-1] / errors_delta[1:])
        / np.log(2.0)
    )

    order_vx = (
        np.log(errors_vx[:-1] / errors_vx[1:])
        / np.log(2.0)
    )

    # Semi-implicit Euler: component-wise convergence orders.
    assert np.all((order_delta > 1.7) & (order_delta < 2.3))
    assert np.all((order_vx > 0.7) & (order_vx < 1.3))