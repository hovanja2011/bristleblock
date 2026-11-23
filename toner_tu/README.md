# Toner–Tu continuum model

This package contains the continuum active-matter model only. It does not
construct mazes and does not implement maze-specific boundary conditions.

- `fields.py`: cell-centred grid and state arrays `rho, vx, vy`.
- `parameters.py`: physical coefficients.
- `domain.py`: interface for spatial operators; periodic reference domain.
- `operators.py`: periodic finite-difference operators.
- `equations.py`: physical Toner–Tu right-hand side.
- `solver.py`: SSP-RK2 integration.
- `initial_conditions.py`, `diagnostics.py`, `visualization.py`: support tools.
- `tests/`: tests for the bulk equations and periodic operators.

The domain interface is the only point where a spatial boundary implementation
enters the physical model. Concrete geometries belong in `../mazes/`.
