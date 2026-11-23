# Maze geometry and boundary operators

This is the third independent project in the repository. It owns physical
domains, wall geometry and the numerical operators needed to impose their
boundary conditions.

The first domain is `BoxMaze`: an empty rectangle with four impermeable
free-slip walls. Cell-centred normal velocity components are odd-reflected
across the corresponding wall; scalar and tangential components are
even-reflected. Density transport is a conservative finite-volume flux
divergence, with external face fluxes set exactly to zero.

The package implements the `toner_tu.domain.DomainOperators` protocol so the
continuum equations do not need to know whether they are running in a box,
corridor or a more complicated maze. The current box Laplacian uses a
five-point stencil and its Neumann-wall accuracy is checked by a grid
refinement test.

Future geometry work belongs here: straight internal walls, passages,
rounded corners, and eventually more accurate boundary representations.
Tests in this project check geometry and boundary behaviour; they do not
duplicate the Toner–Tu bulk-equation tests.

Install after the continuum package:

```bash
python -m pip install -e ../toner_tu
python -m pip install -e .
```
