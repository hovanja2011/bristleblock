# Bristlebot and active-matter research code

The repository is split into three projects with deliberately separate roles:

1. **bristlebots/** — discrete robot mechanics, contacts, and particle-level experiments.
2. **toner_tu/** — continuum Toner–Tu active-matter fields, equations, and time integration.
3. **mazes/** — spatial domains, maze geometry, and boundary-specific numerical operators.

The Toner–Tu equation implementation depends on a small `DomainOperators`
interface. This keeps physical constitutive equations independent from the
shape of the computational domain. The `mazes` project supplies concrete
implementations; its first geometry is a rectangular closed box.

Install the Python packages in editable mode:

```bash
python -m pip install -e ./toner_tu
python -m pip install -e ./mazes
python -m pip install -e ./bristlebots
```

Run the tests for each project from the repository root:

```bash
python -m pytest bristlebots/tests
python -m pytest toner_tu/tests
python -m pytest mazes/tests
```

See [ARCHITECTURE.md](ARCHITECTURE.md) and the README inside each project for
module responsibilities and design boundaries.
