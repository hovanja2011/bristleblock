# Bristlebot and active-matter research code

The repository contains three cooperating Python projects:

- **bristlebots/** — discrete rigid-body dynamics, propulsion and contact mechanics.
- **toner_tu/** — continuum Toner–Tu active-matter equations and time integration.
- **mazes/** — rectangular domains, internal obstacles and boundary-aware spatial operators.

## Install

Use Python 3.10 or newer, from the repository root:

~~~bash
python -m pip install -e "./toner_tu[viz,test]"
python -m pip install -e "./mazes"
python -m pip install -e "./bristlebots[test]"
~~~

## Test

~~~bash
python -m pytest toner_tu/tests
python -m pytest mazes/tests
python -m pytest bristlebots/tests
~~~

## Run models and view experiments

Run the full validation and visualization suite:

~~~bash
python experiments/run_validation_suite.py
~~~

This opens Matplotlib plots and interactive animations. The bristlebots contact
study displays stiffness/friction sweeps; the nonlinear Toner–Tu run animates
density, velocity and vorticity. The maze experiment plots rounded obstacles
and the geometric grid-refinement error.

For a non-interactive run that only writes files:

~~~bash
python experiments/run_validation_suite.py --no-show
~~~

All CSV tables and PNG figures are written to results/validation/. The combined
summary is summary.csv. GitHub Actions uploads the same directory as the
validation-results artifact.

Run one experiment by itself:

~~~bash
python bristlebots/experiments/contact_parameter_sweep.py
python toner_tu/experiments/nonlinear_spatial_dynamics.py
python mazes/experiments/rounded_obstacle_validation.py
~~~

Every experiment accepts --output-dir PATH and --no-show.

## Rounded maze geometry example

~~~python
from mazes import MazeGeometry, CircleObstacle, RoundedPolyline

maze = MazeGeometry(
    width=20.0, height=12.0,
    obstacles=(
        RoundedPolyline(((4.0, 3.0), (12.0, 3.0), (12.0, 9.0)), radius=0.25),
        CircleObstacle((16.0, 7.0), radius=1.0),
    ),
)
grid = maze.grid(200, 120)
operators = maze.operators(grid)
fields = operators.mask_fields(fields)
result = integrate(fields, grid, parameters, dt=1e-3, n_steps=1000,
                   domain=operators)
~~~

The maze boundary operators conserve density and block flux through internal
solid faces. Rounded shapes are analytically defined but currently represented
by a grid-resolved Cartesian mask, not a high-order cut-cell boundary method.

See ARCHITECTURE.md, STAGE1_AUDIT.md, and STAGE2_VALIDATION.md for model
assumptions, numerical scope and known limitations.
