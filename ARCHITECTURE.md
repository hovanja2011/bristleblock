# Repository architecture

This repository is organized as three independent Python projects:

- `bristlebots/` — discrete bristlebot mechanics, contact models and experiments.
- `toner_tu/` — continuum Toner–Tu fields, physical parameters, equations and time integration.
- `mazes/` — domain geometry and boundary-specific spatial operators.

Install in editable mode:

```bash
python -m pip install -e ./toner_tu
python -m pip install -e ./mazes
```

The continuum equations depend on the `DomainOperators` protocol. Geometry
and boundary implementations belong in `mazes/`, not in the Toner–Tu model.
The first implementation is an empty rectangular box with four impermeable
free-slip walls. Future corridors, internal obstacles and rounded walls should
be added as new maze geometries using the same operator interface.