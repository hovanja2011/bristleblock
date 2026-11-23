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


The propulsion force is applied at the body-fixed point `propulsion_offset`,
measured from the centre of mass. Its moment `r × F` is included in the
rotational equation; `active_torque` represents any additional independent
torque.

Contact is represented by a compliant penalty law (linear normal
spring-dashpot and Cundall–Strack tangential friction). Consequently, small
finite overlaps are part of the numerical contact approximation; they are not
an exact hard-contact constraint. Results involving collisions must therefore
be checked for time-step and stiffness convergence.
