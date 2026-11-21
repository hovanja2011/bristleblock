
import numpy as np

from toner_tu.fields import Fields, Grid
from toner_tu.parameters import TonerTuParameters
from toner_tu.boundary_conditions import ChannelBoundaryConditions
from toner_tu.channel_solver import integrate


def restrict_by_averaging(field):
    """Average each 2x2 block of fine cells onto the coarse grid."""
    ny, nx = field.shape

    if ny % 2 != 0 or nx % 2 != 0:
        raise ValueError("Field dimensions must be divisible by 2.")

    return field.reshape(
        ny // 2, 2, nx // 2, 2
    ).mean(axis=(1, 3))


def l2_error(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def observed_order(error_coarse, error_fine):
    if error_coarse <= 0.0 or error_fine <= 0.0:
        raise ValueError("Errors must be positive.")

    return float(np.log2(error_coarse / error_fine))


def test_channel_solver_spatial_convergence():
    params = TonerTuParameters(
        alpha=1.0,
        beta=1.0,
        sigma=0.1,
        diffusion=0.05,
        lambda_=0.5,
    )

    boundary = ChannelBoundaryConditions(
        rho_in=1.0,
        vx_in=0.5,
        vy_in=0.0,
    )

    final_time = 0.005
    dt = 0.0001

    solutions = []

    for nx, ny in [(32, 16), (64, 32), (128, 64)]:
        grid = Grid(
            nx=nx,
            ny=ny,
            lx=20.0,
            ly=10.0,
        )

        x, y = grid.coordinates()

        rho = 1.0 + 0.05 * np.sin(2.0 * np.pi * x / grid.lx)
        vx = 0.5 + 0.02 * np.sin(2.0 * np.pi * x / grid.lx)
        vy = 0.01 * np.sin(2.0 * np.pi * y / grid.ly)

        initial = Fields(rho=rho, vx=vx, vy=vy)

        result = integrate(
            initial,
            grid,
            params,
            boundary,
            dt,
            round(final_time / dt),
        )

        solutions.append(result)

    coarse, medium, fine = solutions

    field_names = ("rho", "vx", "vy")

    for field_name in field_names:
        coarse_field = getattr(coarse, field_name)
        medium_field = getattr(medium, field_name)
        fine_field = getattr(fine, field_name)

        # Align cell centers by averaging 2x2 fine-cell blocks.
        medium_on_coarse = restrict_by_averaging(medium_field)
        fine_on_medium = restrict_by_averaging(fine_field)

        error_coarse = l2_error(
            coarse_field,
            medium_on_coarse,
        )

        error_fine = l2_error(
            medium_field,
            fine_on_medium,
        )

        order = observed_order(error_coarse, error_fine)

        assert np.isfinite(order)
        assert error_fine < error_coarse, (
            f"{field_name}: spatial error did not decrease "
            f"under refinement: "
            f"coarse error={error_coarse}, "
            f"fine error={error_fine}, "
            f"observed order={order}"
        )
        assert order > 0.0
