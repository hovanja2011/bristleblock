# Stage 2 — reproducible analytical validation experiments

Status: initial benchmark pair added; CI validation pending.

## Purpose

Stage 2 turns selected analytical checks into reproducible experiments with
fixed physical parameters and initial conditions. Each benchmark reports
quantitative errors and refinement behaviour. It does not introduce new
physical parameters fitted to the numerical output.

## Implemented benchmark pair

### Bristlebots: free propulsion with linear drag

The reference problem is
\[
m\dot v_x = F-\gamma v_x,\qquad
m\dot v_y = -\gamma v_y.
\]
The exact velocity is exponential and the exact position follows by
integration. The numerical update is compared at matched physical times.
Both position and velocity errors are measured for successively halved
time steps. The current integrator is semi-implicit Euler, so first-order
global convergence is expected.

Run:
\`\`\`bash
python bristlebots/experiments/free_motion_validation.py
\`\`\`

### Toner–Tu: periodic linear acoustic Fourier mode

A small-amplitude density perturbation and zero initial velocity evolve as a
periodic acoustic mode. The numerical solution is compared with the
continuum solution at fixed physical time on successively refined grids.
The time step is reduced as \(\Delta t\propto \Delta x^2\), making the
third-order time-integration error subdominant to the second-order centred
spatial error.

Run:
\`\`\`bash
python toner_tu/experiments/periodic_acoustic_convergence.py
\`\`\`

## Scope and limitations

- These are reference problems, not validation against experimental bristlebots data.
- The Toner–Tu acoustic benchmark validates a linearized, small-amplitude
  regime; it does not establish convergence of the full nonlinear model.
- The bristlebots benchmark validates free translation with linear drag,
  not contact dynamics or propulsion calibration.
- Maze geometry remains the rectangular closed box. Internal obstacles and
  rounded walls belong to a later geometry stage.
