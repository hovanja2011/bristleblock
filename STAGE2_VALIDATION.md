# Stage 2 — reproducible verification and visualization

Status: the full validation suite passes in GitHub Actions. Run #22 produced the numerical results summarized below. Run #23 reran all three test groups and all seven quantitative experiment scripts, then uploaded the CSV tables and PNG figures as the validation-results workflow artifact.

## Keep three kinds of evidence separate

1. **Code verification:** tests establish that implementation behaves as specified.
2. **Numerical verification:** refinement studies quantify discretization error and observed convergence for a defined problem.
3. **Experimental validation:** comparison with physical bristlebots data; this repository does not yet provide that dataset.

## Bristlebots benchmarks

### Free translation with linear drag
The free-motion experiment compares position and velocity with the exact solution of
\(m\dot v_x=F-\gamma v_x\), \(m\dot v_y=-\gamma v_y\).
The semi-implicit Euler integrator is expected to converge globally at first order.

### Active rotation with rotational drag
The rotational experiment compares angular velocity and angle with the exact solution of
\(I\dot\omega=\tau-\gamma_r\omega\), \(\dot\theta=\omega\).
The propulsion force acts through the centre of mass; this benchmark isolates the independent active torque.

### Head-on damped collision
The head-on experiment begins at first contact with zero overlap and no tangential friction. The normal overlap obeys
\(m_{\rm eff}\ddot\delta+c_n\dot\delta+k_n\delta=0\), with \(m_{\rm eff}=m/2\).
It compares overlap, overlap rate and centre positions with the underdamped analytical solution before separation.

### Oblique frictionless collision
The oblique experiment compares final velocities with the exact elastic hard-disk collision map and reports linear momentum, kinetic energy, angular momentum and maximum penalty overlap. The compliant finite-stiffness trajectory is not an exact hard-contact trajectory; the output is a time-step refinement and conservation diagnostic at fixed stiffness.

## Toner–Tu benchmarks

### Periodic undamped acoustic mode
The periodic acoustic experiment compares a small-amplitude Fourier mode with the continuum linear acoustic solution.

### Periodic damped acoustic mode
The damped acoustic experiment includes velocity diffusion. Reference amplitudes are obtained from the exact matrix exponential of the continuum linearized Fourier-mode system. The finite-difference solver is compared at a fixed physical time over grid refinement.

### Nonlinear homogeneous ordering
The nonlinear ordering experiment isolates the full cubic local term
\(\dot v=(\alpha-\beta |v|^2)v\) in a spatially uniform state. The exact solution is logistic in \(|v|^2\). This is a nonlinear local-saturation benchmark, **not** a test of nonlinear spatial pattern formation, vortices or nonlinear transport.

## Results from CI run #22

The following are measured results from the reproducible run, not estimates:

| Benchmark | Refinement result |
|---|---|
| Free translation | Position and velocity errors decrease by approximately 2 when dt is halved (first order). |
| Active rotation | Angle and angular-velocity errors show order 1.00–1.02. |
| Head-on damped collision | Overlap and position errors show order about 1.05–1.27; overlap-rate error is approximately first order. |
| Oblique frictionless collision | Final-velocity error falls from \(7.95\\times10^{-5}\) to \(9.89\\times10^{-7}\) over an 8-fold dt reduction; total momentum and angular-momentum drifts remain near floating-point roundoff. |
| Undamped acoustic mode | Spatial order 1.99–2.00. |
| Damped acoustic mode | Spatial order 2.00. |
| Nonlinear homogeneous ordering | Time order 2.97–3.05, consistent with SSP-RK3. |

**Contact-overlap warning:** in the oblique-collision run, maximum overlap is approximately \(1.581\\times10^{-2}\), or 3.16% of the robot radius, at the chosen normal stiffness \(k_n=2000\). It remains nearly constant under dt refinement. This is expected: reducing dt controls temporal error but does not remove finite penalty deformation. A separate stiffness-refinement study is required before treating the compliant contact as a sufficiently accurate hard-disk approximation.

## Reproducibility and outputs

From the repository root:

~~~bash
python -m pip install -e "./toner_tu[viz]"
python -m pip install -e "./mazes"
python -m pip install -e "./bristlebots"
python experiments/run_validation_suite.py
~~~

Use --no-show for headless runs. Results are written to results/validation:
CSV error tables, PNG plots, and a combined summary.csv. Individual scripts
accept --output-dir PATH and --no-show. GitHub Actions uploads the same output
directory as the validation-results artifact after a successful run.

The interactive animation is generated from sampled numerical states; the
quantitative plots overlay the analytical solution or clearly identified
reference. Animations are qualitative visualizations; error tables are the
quantitative comparison.

## Limitations and interpretation

- Translation, rotation and head-on normal-contact benchmarks compare with continuous analytical solutions; semi-implicit Euler shows first-order global convergence.
- The oblique-collision benchmark uses the ideal elastic collision map as a final-state reference and separately monitors conserved quantities. It does not claim that finite penalty stiffness reproduces a hard-contact path exactly.
- The Toner–Tu damped-mode benchmark checks a linearized Fourier mode with a small perturbation. The nonlinear benchmark checks only homogeneous cubic saturation.
- Nonlinear spatial Toner–Tu regimes still need separate verification, ideally with manufactured solutions or a trusted independent high-resolution reference.
- Maze geometry remains a rectangular closed box. Internal obstacles and rounded walls belong to a later geometry stage.
- None of these benchmarks constitutes validation against physical bristlebots experiments.
