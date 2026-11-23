# Bristlebots

This project contains the discrete mechanical model of individual bristlebot
robots and their interactions.

- `src/bristlebots/robot.py`: robot state, geometry and propulsion parameters.
- `contacts.py`: robot-wall and robot-robot contact forces and friction state.
- `simulation.py`: time integration of one or multiple robots.
- `visualization.py`: trajectory and animation helpers.
- `experiments/`: reproducible physical/numerical demonstrations.
- `tests/`: analytical references, convergence and collision regression tests.

This package is intentionally separate from the continuum Toner–Tu model.
