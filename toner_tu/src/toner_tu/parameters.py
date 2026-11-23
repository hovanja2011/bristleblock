"""Physical coefficients of the simplified polar Toner–Tu model."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TonerTuParameters:
    """Model coefficients: local ordering, saturation, density coupling and diffusion."""
    alpha: float
    beta: float
    sigma: float
    diffusion: float
    lambda_: float = 1.0

    def __post_init__(self) -> None:
        values = np.asarray([self.alpha, self.beta, self.sigma, self.diffusion, self.lambda_], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("All Toner–Tu parameters must be finite.")
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
