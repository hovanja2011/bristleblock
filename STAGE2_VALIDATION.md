# Numerical verification and geometry validation

## Evidence levels

1. **Code verification:** automated tests establish implementation properties.
2. **Numerical verification:** refinement and conservation studies quantify error for specified equations, geometries and parameters.
3. **Experimental validation:** comparison with physical bristlebots data. Such a dataset is not yet part of this repository.

## Bristlebots

The free translation, active rotation and head-on damped-contact benchmarks compare against continuous analytical solutions. The oblique frictionless collision compares final velocities with the ideal elastic hard-disk collision map; the compliant trajectory is not claimed to be identical to a hard-contact path.

The contact-parameter sweep varies normal stiffness and static/kinetic friction. It reports maximum overlap, normal/tangential impulse, contact mode counts, final spin, energy loss, momentum/angular-momentum drift and the change under a halved time step. The stiffness study is essential because reducing dt does not remove penalty deformation. Frictional oblique contact has no single closed-form trajectory for the implemented history-dependent Cundall–Strack law; these are controlled parameter studies, not analytical benchmarks.

## Toner–Tu

The linear acoustic and damped Fourier-mode experiments use analytical continuum references. The homogeneous cubic ordering experiment has an exact logistic speed solution, but it does not test spatial nonlinearities.

The nonlinear spatial experiment starts from seeded random velocity perturbations and evolves the full periodic Toner–Tu equations. It records density, polarization, mean speed, vorticity RMS and enstrophy, and compares final fields across time-step refinements against the finest run. Interactive animations display density, velocity arrows and vorticity. This is a nonlinear numerical convergence study, not an analytical solution. Vortex emergence depends on the parameter regime and should be judged from the recorded fields and diagnostics rather than assumed from the script name.

## Maze geometry and internal walls

MazeGeometry supports circular obstacles and finite-thickness rounded segments (capsules with circular end caps). MazeDomainOperators blocks density flux across every face adjacent to a solid cell, including internal obstacles, and conserves the discrete total mass. The validation experiment compares the represented solid area with the exact area of the circle-plus-capsule geometry over grid refinement and measures mass drift during transport.

Important limitation: obstacle shapes are defined analytically by signed-distance functions but sampled on a Cartesian cell-centred mask. Thus the boundary is staircase-resolved at the grid scale; this is not a cut-cell, immersed-boundary or high-order curved-wall method. Velocity diffusion uses a free-slip-like zero-normal-gradient mask closure. Results close to curved boundaries require grid-refinement checks.

## Reproducible commands

Install all three projects:

~~~bash
python -m pip install -e "./toner_tu[viz,test]"
python -m pip install -e "./mazes"
python -m pip install -e "./bristlebots[test]"
~~~

Run tests:

~~~bash
python -m pytest toner_tu/tests
python -m pytest mazes/tests
python -m pytest bristlebots/tests
~~~

Run the full experiment suite with interactive plots and animations:

~~~bash
python experiments/run_validation_suite.py
~~~

Use --no-show for headless runs. CSV and PNG outputs are written to results/validation. Individual scripts accept --output-dir PATH and --no-show.

## Interpretation

The new checks close the requested numerical-verification gaps for contact parameter dependence, nonlinear spatial evolution and rounded internal obstacle masks. They do not constitute experimental validation of the bristlebots model, do not prove convergence of Toner–Tu vortex statistics in the thermodynamic limit, and do not yet provide sub-grid accurate curved-wall hydrodynamics.
