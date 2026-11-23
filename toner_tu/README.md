# Toner–Tu continuum model

This package contains the continuum active-matter model only. It does not
construct mazes and does not implement maze-specific boundary conditions.

- `fields.py`: cell-centred Cartesian grid and state arrays `rho, vx, vy`.
- `parameters.py`: validated physical coefficients.
- `domain.py`: protocol for domain-dependent spatial operators and periodic reference domain.
- `operators.py`: periodic finite-difference operators.
- `equations.py`: physical Toner–Tu right-hand side, independent of geometry.
- `solver.py`: SSP-RK2 integration and a geometry-independent explicit stability estimate.
- `initial_conditions.py`, `diagnostics.py`, `visualization.py`: support tools.
- `tests/`: tests for bulk equations, periodic operators, and time integration.

The time-step estimate includes advection, the linearized density-pressure
coupling, velocity diffusion and the local cubic ordering term. The
`step()` function rejects a step above its estimated limit. The `integrate()`
function treats `dt` as the requested output interval and adaptively divides
it into stable substeps, recalculating the limit as the state evolves.

The domain interface is the only point where a spatial boundary implementation
enters the physical model. Concrete geometries belong in `../mazes/`.
