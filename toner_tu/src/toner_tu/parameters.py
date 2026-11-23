"""Physical coefficients of the simplified polar Toner–Tu model."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TonerTuParameters:
    """Model coefficients: local ordering, saturation, density coupling and diffusion."""
    alpha: float
    beta: float
    sigma: float
    diffusion: float
    lambda_: float = 1.0

    def __post_init__(self) -> None:
        if self.beta <= 0:
            raise ValueError("beta must be positive.")
        if self.sigma < 0:
            raise ValueError("sigma must be non-negative.")
        if self.diffusion < 0:
            raise ValueError("diffusion must be non-negative.")

    @property
    def ordered_speed(self) -> float:
        """Homogeneous ordered-state speed sqrt(alpha/beta), or zero if disordered."""
        if self.alpha <= 0:
            return 0.0
        return (self.alpha / self.beta) ** 0.5
