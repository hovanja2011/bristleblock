# Maze geometry and boundary operators

The maze package owns physical domains, internal obstacles and boundary-specific
numerical operators. It provides BoxMaze, CircleObstacle, RoundedSegment
(finite-thickness segment with circular end caps), MazeGeometry, and
MazeDomainOperators.

Example:

~~~python
from mazes import MazeGeometry, CircleObstacle, RoundedSegment

maze = MazeGeometry(
    width=20.0, height=12.0,
    obstacles=(
        RoundedSegment((4.0, 3.0), (16.0, 3.0), radius=0.25),
        CircleObstacle((10.0, 7.0), radius=1.0),
    ),
)
grid = maze.grid(200, 120)
operators = maze.operators(grid)
fields = operators.mask_fields(fields)
result = integrate(fields, grid, parameters, dt=1e-3, n_steps=1000,
                   domain=operators)
~~~

Obstacle shapes are analytic signed-distance functions, but the current
operator discretization is a Cartesian cell-centre mask. Interfaces are
therefore resolved at grid scale, not by cut cells. Density flux through every
face adjacent to a solid cell is exactly zero, so the discrete continuity
equation is conservative. Velocity diffusion uses a zero-normal-gradient,
free-slip-like mask closure. This is a first geometry-capable solver, not yet
a high-order curved-wall scheme. Grid refinement is required for quantitative
results near rounded boundaries.

Install after the continuum package:

~~~bash
python -m pip install -e ../toner_tu
python -m pip install -e .
~~~
