# Stage 1 — rectangular-domain physics and numerics audit

Status: implemented and CI-verified on branch `stage1-physics-audit`.

## Scope

Stage 1 is restricted to the two-dimensional rectangular closed box and the
bulk periodic reference problem. It does not include internal maze walls,
rounded corners, reconfigurable maze cells, or percolation geometries.

## Implemented checks and decisions

### Bristlebots

- The active propulsion force acts through the centre of mass.
- `active_torque` remains a separately prescribed torque; no moment from an
  off-centre propulsion force is included.
- The unsupported `propulsion_offset` and derived `propulsion_torque`
  mechanisms were removed from the stage branch.
- Contact tests compare normal spring and damped spring-dashpot collisions
  against analytic solutions, check friction limits and invariants, and check
  time-step refinement.
- The robot model uses finite-stiffness penalty contacts. Small overlap is
  therefore a model property, not exact rigid-body non-penetration.

### Toner–Tu solver

- The time integrator is SSP-RK3. The previous SSP-RK2 scheme is unsuitable
  for nonzero purely imaginary eigenvalues of centred periodic acoustic modes.
- The explicit time-step controller includes estimates for advection,
  acoustic propagation, diffusion and the local ordering reaction; it checks
  intermediate RK states and adaptively retries rejected substeps.
- The estimate is a conservative engineering safeguard for the implemented
  operators, not a proof of nonlinear stability for arbitrary future
  discretizations or geometries.
- Tests cover uniform stationary/ordered states, mass conservation,
  adaptive substepping, a periodic acoustic-mode energy check, and stability
  bound response to velocity and grid refinement.

### Rectangular box and wall conditions

- The box uses cell-centred fields, even reflection for scalar/tangential
  quantities and odd reflection for wall-normal velocity.
- External normal face velocities and external density fluxes are set to
  exactly zero.
- Tests cover mass conservation over integration, zero net density flux,
  exact zero normal wall-face velocity, and second-order convergence of the
  Neumann Laplacian.
- The conservative upwind density flux is checked for spatial convergence.
- A coupled small-amplitude acoustic standing wave with impermeable
  Neumann walls is compared with its analytic solution on successively
  refined grids. Its convergence is expected to be first order because the
  box density transport uses an upwind flux.

## Validation record

GitHub Actions workflow run
[45134256353](https://github.com/hovanja2011/bristleblock/actions/runs/45134256353)
passed before the final analytic box-wave convergence test was added:
33 Toner–Tu tests, 8 maze tests and 38 bristlebots tests (79 total).
The final test suite must be re-run after this document and test are committed.

## Explicitly deferred

- Grid/time convergence of the full nonlinear Toner–Tu model across a broad
  parameter regime.
- Spectral proof of the time-step estimate for every domain operator.
- Rigid-body contact without penalty overlap.
- Internal walls, rounded maze geometry, and maze topology.
