# Maze geometry and boundary operators

This is the third independent project in the repository. It owns physical
domains, wall geometry and the numerical operators needed to impose their
boundary conditions.

The first domain is `BoxMaze`: an empty rectangle with four impermeable
free-slip walls. The package implements the `toner_tu.domain.DomainOperators`
protocol so the continuum equations do not need to know whether they are
running in a box, corridor or a more complicated maze.

Future geometry work belongs here: straight internal walls, passages,
rounded corners, and eventually more accurate boundary representations.
Tests in this project check geometry and boundary behaviour; they do not
duplicate the Toner–Tu bulk-equation tests.

Install after the continuum package:

```bash
python -m pip install -e ../toner_tu
python -m pip install -e .
```
