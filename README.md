# Bristlebot and active-matter research code

The repository is split into three projects with deliberately separate roles:

1. **bristlebots/** — discrete robot mechanics, contacts, and particle-level experiments.
2. **toner_tu/** — continuum Toner–Tu active-matter fields, equations, and time integration.
3. **mazes/** — spatial domains, maze geometry, and boundary-specific numerical operators.

The Toner–Tu equation implementation depends on a small DomainOperators
interface. This keeps physical constitutive equations independent from the
shape of the computational domain. The mazes project supplies concrete
implementations; its first geometry is a rectangular closed box.

## Installation

Use Python 3.10 or newer. From the repository root:

~~~bash
python -m pip install -e "./toner_tu[viz]"
python -m pip install -e "./mazes"
python -m pip install -e "./bristlebots"
~~~

The viz extra installs Matplotlib for plots and animations. The bristlebots
package already depends on Matplotlib.

## Run tests

~~~bash
python -m pytest bristlebots/tests
python -m pytest toner_tu/tests
python -m pytest mazes/tests
~~~

## Run and visualize the models

Run the full reproducible validation suite with interactive plots and
animations:

~~~bash
python experiments/run_validation_suite.py
~~~

For a headless run that writes tables and PNG figures without opening windows:

~~~bash
python experiments/run_validation_suite.py --no-show
~~~

Generated results are written to results/validation/: per-experiment CSV
tables, PNG plots, a combined summary.csv, and a short README describing the
artifacts. A successful GitHub Actions run also publishes the folder as the
validation-results artifact, available from the run's Artifacts section.

Individual scripts can also be run separately:

~~~bash
python bristlebots/experiments/free_motion_validation.py
python bristlebots/experiments/rotational_validation.py
python bristlebots/experiments/head_on_collision_validation.py
python bristlebots/experiments/oblique_collision_validation.py
python toner_tu/experiments/periodic_acoustic_convergence.py
python toner_tu/experiments/damped_acoustic_mode.py
python toner_tu/experiments/nonlinear_ordering_validation.py
~~~

Interactive experiments open Matplotlib figures; the bristlebots rotation and
collision experiments and the Toner–Tu acoustic modes also show animations.
Every script accepts --output-dir PATH and --no-show.

The numerical curves are compared with analytical solutions or explicitly
identified hard-contact reference limits. Convergence tables report errors
and observed orders; conservation diagnostics are reported separately.
Successful benchmark tests validate only their stated setup and parameter
regime, not the entire physical model or agreement with laboratory data.

See ARCHITECTURE.md, STAGE1_AUDIT.md, and STAGE2_VALIDATION.md for design,
scope, assumptions, and interpretation.
