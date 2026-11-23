# Numerical verification and geometry validation

## Evidence levels

1. **Code verification:** automated tests establish implementation properties.
2. **Numerical verification:** refinement and conservation studies quantify error for specified equations, geometries and parameters.
3. **Experimental validation:** comparison with physical bristlebots data. Such a dataset is not yet part of this repository.

## Bristlebots

The free translation, active rotation and head-on damped-contact benchmarks compare against continuous analytical solutions. The oblique frictionless collision compares final velocities with the ideal elastic hard-disk collision map; the compliant trajectory is not claimed to be identical to a hard-contact path.

### Contact stiffness and friction

The oblique-contact sweep varies normal stiffness \(k_n=500\ldots4000\) and static friction \(\mu_s=0\ldots0.6\), with \(\mu_k=0.8\mu_s\). It reports maximum overlap, normal/tangential impulse, contact mode counts, final spin, energy loss, momentum/angular-momentum drift and sensitivity to halving the time step.

The sweep showed maximum overlap decreasing from 2.29% to 0.815% of the robot radius as \(k_n\) increased from 500 to 4000. Friction produced measurable angular velocity changes. The sweep is a parameter study, not an analytical solution of the history-dependent Cundall–Strack friction law.

The sweep exposed a contact-geometry issue: using each full disk radius as the friction lever arm during finite overlap created a small artificial couple. Robot–robot contact now uses one shared midpoint between the closest surface points; at zero overlap this reduces to the physical surface contact point. The independent high-accuracy sticking-contact ODE reference was updated to the same geometry. The frictional pair tests now verify linear and angular momentum conservation.

## Toner–Tu

The linear acoustic and damped Fourier-mode experiments use analytical continuum references. The homogeneous cubic ordering experiment has an exact logistic speed solution, but it does not test spatial nonlinearities.

### Nonlinear spatial evolution and vorticity

The nonlinear spatial experiment starts from smooth, seeded long-wavelength velocity perturbations around a weakly ordered state and evolves the full periodic Toner–Tu equations. It records density, polarization, mean speed, vorticity RMS and enstrophy, and compares final fields across time-step refinements against the finest run. Interactive animations display density, velocity arrows and vorticity.

In the current reproducible parameter set, vorticity RMS grew from \(4.05\times10^{-2}\) to \(1.00\times10^{-1}\) over the run (factor 2.48); enstrophy grew from \(8.22\times10^{-4}\) to \(5.04\times10^{-3}\). Global polarization at the final time was 0.262. This is evidence of amplification of spatially structured vorticity in this chosen numerical regime, not an analytical solution or proof of a universal vortex-forming instability. The time-refinement comparison is against the finest numerical run.

## Maze geometry and internal walls

MazeGeometry supports circular obstacles, finite-thickness rounded segments (capsules) and rounded polylines. A RoundedPolyline is the union of capsules along a centreline, so a bend is thickened by a disk instead of retaining a sharp zero-radius corner.

MazeDomainOperators blocks density flux across every face adjacent to a solid cell and also checks the signed distance at face centres. This prevents flow through a thin obstacle even when both neighbouring cell centres happen to lie in fluid. The discrete continuity equation is conservative; the validation run measured relative mass drift of order \(10^{-15}\).

Important limitation: obstacle shapes are defined analytically by signed-distance functions but sampled on a Cartesian cell-centred mask. Thus the boundary is staircase-resolved at the grid scale; this is not a cut-cell, immersed-boundary or high-order curved-wall method. Velocity diffusion uses a free-slip-like zero-normal-gradient mask closure. Results close to curved boundaries require grid-refinement checks. The solid-area error can be non-monotone at intermediate resolutions because of cell-centre alignment; use several refinements rather than interpreting one grid comparison as proof of formal order.

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

Use --no-show for headless runs. CSV and PNG outputs are written to results/validation. Individual scripts accept --output-dir PATH and --no-show. GitHub Actions uploads the complete output directory as the validation-results artifact.

## Interpretation and remaining limits

These additions close the requested numerical-verification gaps for contact parameter dependence, nonlinear spatial evolution and rounded internal obstacle masks. They do not constitute experimental validation of the bristlebots model, do not prove convergence of Toner–Tu vortex statistics in the thermodynamic limit, and do not yet provide sub-grid accurate curved-wall hydrodynamics.
