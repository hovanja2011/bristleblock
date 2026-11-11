# Bristlebot dynamics simulator

## 1. Purpose and scope

This project implements a two-dimensional microscopic model of bristlebot-like robots. The current implementation describes a robot as a rigid circular disk with translational and rotational degrees of freedom. The robot can propel itself, experience linear drag, and interact mechanically with fixed wall segments.

The immediate purpose of the model is to provide a controllable microscopic simulator that can later be extended to multiple robots and maze-like environments. Its outputs can be used to study trajectories, velocities, angular motion, and spatial distributions. A later research stage may compare coarse-grained fields from the microscopic model with continuum descriptions such as the Toner–Tu equations. That comparison is a research objective, not a feature implemented by the present code.

### Implemented features

- Two-dimensional rigid-disk robot state.
- Constant-magnitude active propulsion along the robot's body axis.
- Linear translational and rotational drag.
- Contact with finite, immobile wall segments.
- Compliant normal contact with optional linear damping.
- Cundall–Strack tangential contact history with static and kinetic Coulomb limits.
- Contact torque and coupling between translation and rotation.
- Semi-implicit Euler time integration.
- Matplotlib animation of robot trajectories, orientations, angular velocities, and walls.
- Unit tests for selected contact and integration properties.

### Not implemented in the supplied code

- Robot–robot contact.
- Moving or deformable walls.
- Three-dimensional motion.
- Hydrodynamic or fluid-mediated interactions.
- A dedicated maze generator or boundary-condition framework.
- A Toner–Tu solver or automated microscopic-to-continuum parameter fitting.
- Rolling resistance or a separate model for the bristle-level mechanics.

The robot is therefore an effective mechanical representation of a bristlebot, not a simulation of individual bristles or their detailed interaction with a substrate.

---

## 2. Robot state and physical parameters

The `Robot` class stores the state of one disk in two dimensions:

| Variable | Meaning |
|---|---|
| `position` | Center-of-mass position \(\mathbf r=(x,y)\) |
| `velocity` | Center-of-mass velocity \(\mathbf v=(v_x,v_y)\) |
| `theta` | Body orientation angle \(\theta\) |
| `omega` | Angular velocity \(\omega=\dot\theta\) |
| `radius` | Disk radius \(R\) |
| `mass` | Mass \(m\) |
| `propulsion_force` | Active-force magnitude \(F_a\) |
| `translational_drag` | Translational drag coefficient \(\zeta_t\) |
| `rotational_drag` | Rotational drag coefficient \(\zeta_r\) |

The model assumes a uniform solid disk. Its moment of inertia about the center, perpendicular to the plane, is

\[
I = \frac{1}{2}mR^2.
\]

The body-axis unit vector is

\[
\mathbf e(\theta)=
\begin{pmatrix}
\cos\theta\\
\sin\theta
\end{pmatrix}.
\]

The `direction` property returns this vector. The active force is applied at the center of mass:

\[
\mathbf F_a=F_a\mathbf e(\theta).
\]

Because the active force is applied at the center, it produces no direct torque in the present model.

The `free_speed` property returns

\[
v_0=\frac{F_a}{\zeta_t}.
\]

This is the steady speed of an isolated robot under constant propulsion and linear translational drag. The expression is defined only for \(\zeta_t>0\). The current property does not guard against a zero drag coefficient; calling it when `translational_drag == 0` causes division by zero. The time integrator itself does not use this property.

The current `Robot` constructor does not validate that mass and radius are positive, or that drag and propulsion parameters are physically admissible. Callers should provide physically meaningful values.

---

## 3. Equations of motion

The robot has two translational degrees of freedom and one rotational degree of freedom.

### 3.1 Translation

The total translational force consists of active propulsion, linear drag, and wall-contact forces:

\[
m\dot{\mathbf v}
=
F_a\mathbf e(\theta)
-\zeta_t\mathbf v
+\sum_j \mathbf F_{c,j}.
\]

Here, \(\mathbf F_{c,j}\) is the contact force exerted by wall \(j\). The sum is taken over the walls passed to `step`.

The kinematic equation is

\[
\dot{\mathbf r}=\mathbf v.
\]

In the absence of walls, the translational equation becomes

\[
m\dot{\mathbf v}
=
F_a\mathbf e(\theta)-\zeta_t\mathbf v.
\]

For fixed orientation and positive drag, the steady velocity is \(v_0\mathbf e(\theta)\), where \(v_0=F_a/\zeta_t\).

### 3.2 Rotation

The rotational equation is

\[
I\dot\omega
=
-\zeta_r\omega
+\sum_j\tau_{c,j},
\]

with

\[
\dot\theta=\omega.
\]

The rotational drag torque is linear in angular velocity. Contact forces can generate torque if their line of action does not pass through the robot center. In the current disk–wall model, the normal force acts along the radius and therefore produces no torque; the tangential force produces the contact torque.

There is no intrinsic active torque or spontaneous chirality in the current implementation.

---

## 4. Wall geometry

The `Wall` class represents a finite, immobile line segment defined by endpoints \(\mathbf a\) and \(\mathbf b\).

Its length and unit direction are

\[
L=\|\mathbf b-\mathbf a\|,
\qquad
\mathbf t_w=\frac{\mathbf b-\mathbf a}{L}.
\]

A zero-length segment is rejected.

### 4.1 Closest point on a finite segment

For a point \(\mathbf p\), the projection parameter is

\[
s=(\mathbf p-\mathbf a)\cdot\mathbf t_w.
\]

It is clamped to the segment:

\[
s_c=\min(\max(s,0),L).
\]

The closest point is

\[
\mathbf q=\mathbf a+s_c\mathbf t_w.
\]

Clamping is important: the wall is a finite segment, not an infinite line. Near an endpoint, the closest point can be the endpoint itself.

### 4.2 Disk–wall contact geometry

For a disk centered at \(\mathbf r\), define

\[
d=\|\mathbf r-\mathbf q\|,
\qquad
\delta=\max(R-d,0).
\]

Here \(d\) is the distance from the disk center to the closest point on the wall, and \(\delta\) is the overlap (penetration depth) used by the compliant contact model.

When \(d\) is nonzero, the unit normal pointing from the closest wall point toward the disk center is

\[
\mathbf n=\frac{\mathbf r-\mathbf q}{d}.
\]

If the center lies exactly on the closest point, the code uses a fallback normal perpendicular to the wall direction:

\[
\mathbf n=
\begin{pmatrix}
-t_{w,y}\\
t_{w,x}
\end{pmatrix}.
\]

The fallback resolves the otherwise undefined direction, but it selects one of the two possible perpendicular directions. The model does not currently store a preferred side of a wall.

---

## 5. Normal contact force

The contact model is compliant: a small overlap is permitted, and the overlap generates a restoring force. This is a soft-contact approximation rather than an exact rigid-body collision constraint.

The normal component of the robot velocity is

\[
v_n=\mathbf v\cdot\mathbf n.
\]

The scalar normal force is calculated as

\[
F_n=
\max\left(0,\;k_n\delta-\eta_n v_n\right),
\]

where:

- \(k_n\) is the normal stiffness;
- \(\eta_n\) is the normal damping coefficient;
- \(\delta\) is the overlap;
- \(v_n\) is the normal velocity.

The vector force on the robot is

\[
\mathbf F_n=F_n\mathbf n.
\]

With the chosen normal convention, an approaching disk has \(v_n<0\), so the damping term increases the repulsive force. If the computed scalar force is non-positive, it is clipped to zero. If there is no overlap, the contact force is zero.

For zero normal damping,

\[
F_n=k_n\delta
\]

while contact is active.

The contact stiffness introduces a characteristic time scale. For a disk of mass \(m\) against an immobile wall, the undamped normal-contact frequency is

\[
\omega_c=\sqrt{\frac{k_n}{m}},
\qquad
T_c=\frac{2\pi}{\omega_c}.
\]

The time step must be sufficiently small compared with the contact time scale to resolve the collision accurately. The appropriate value should be established by convergence tests for the chosen stiffness and masses; it should not be assumed universal.

---

## 6. Tangential friction: Cundall–Strack model

The tangential contact model stores a history variable representing accumulated tangential displacement. This is the Cundall–Strack spring-slider construction.

### 6.1 Contact-point velocity

The vector from the disk center to the contact point is

\[
\mathbf r_c=-R\mathbf n.
\]

For a planar rigid body, the velocity of that point is

\[
\mathbf v_c
=
\mathbf v+\omega\,\hat{\mathbf z}\times\mathbf r_c,
\]

or, in components,

\[
\mathbf v_c
=
\mathbf v+
\omega
\begin{pmatrix}
-r_{c,y}\\
r_{c,x}
\end{pmatrix}.
\]

The unit tangent used by the implementation is

\[
\mathbf t=
\begin{pmatrix}
-n_y\\
n_x
\end{pmatrix}.
\]

The tangential contact velocity is

\[
v_t=\mathbf v_c\cdot\mathbf t.
\]

This includes both center-of-mass translation and surface velocity due to rotation.

### 6.2 Tangential displacement history

For each robot–wall pair, the code stores a scalar tangential displacement \(\xi\). It is advanced explicitly:

\[
\xi^{\,new}=\xi^{\,old}+v_t\Delta t.
\]

The trial tangential force is

\[
F_t^{trial}=-k_t\xi-\eta_t v_t,
\]

where \(k_t\) and \(\eta_t\) are the tangential stiffness and damping coefficients.

The corresponding trial vector is \(F_t^{trial}\mathbf t\).

### 6.3 Static-friction branch

The static Coulomb threshold is

\[
F_{s,\max}=\mu_sF_n.
\]

If

\[
|F_t^{trial}|\leq\mu_sF_n,
\]

the implementation retains the trial force:

\[
F_t=F_t^{trial}.
\]

This is a compliant approximation to sticking. It allows a small elastic tangential displacement while preventing the force from exceeding the static Coulomb limit.

### 6.4 Kinetic-friction branch

If the trial force exceeds the static threshold, the code switches to the kinetic branch. For nonzero tangential velocity, the force opposes the direction of slip:

\[
F_t=-\mu_kF_n\,\operatorname{sgn}(v_t).
\]

If the tangential velocity is numerically zero, the implementation uses the sign of the trial force as a fallback for selecting the direction.

After entering the kinetic branch, the stored tangential displacement is corrected so that the spring contribution is consistent with the capped force:

\[
\xi=
\frac{-F_t-\eta_t v_t}{k_t}.
\]

The parameter validation enforces

\[
\mu_s\geq\mu_k\geq0.
\]

Both stiffnesses must be positive, and both damping coefficients must be non-negative.

### 6.5 Total contact force and torque

The total force exerted by one wall is

\[
\mathbf F_c=F_n\mathbf n+F_t\mathbf t.
\]

The contact torque about the robot center is the planar cross product

\[
\tau_c=(\mathbf r_c\times F_t\mathbf t)_z
      =r_{c,x}(F_t\mathbf t)_y-r_{c,y}(F_t\mathbf t)_x.
\]

The normal force gives no torque because it is collinear with \(\mathbf r_c\).

### 6.6 Contact-history handling and limitations

The code resets \(\xi\) to zero when the overlap becomes zero or when the normal force is clipped to zero. Each wall has a separate `ContactState`, stored in the robot's `wall_contact_states` dictionary.

The current implementation uses a scalar history variable in the instantaneous tangent direction. It does not explicitly rotate or project the stored tangential displacement when the contact normal changes. This is a simplification that may matter for curved paths around wall endpoints or rapidly changing contact geometry. A more complete implementation would transport the tangential history consistently as the contact frame rotates.

The model also does not include rolling resistance. The static branch is a spring-based approximation, not an exact constraint solver for rigid static friction.

---

## 7. Numerical integration

The `step` function advances one robot by one time step. It first calculates free-motion forces and torques, then adds the contributions from all supplied walls.

The update is semi-implicit (symplectic) Euler. For translation:

\[
\mathbf a^n=\frac{\mathbf F^n}{m},
\]

\[
\mathbf v^{n+1}=\mathbf v^n+\mathbf a^n\Delta t,
\]

\[
\mathbf r^{n+1}=\mathbf r^n+\mathbf v^{n+1}\Delta t.
\]

For rotation:

\[
\alpha^n=\frac{\tau^n}{I},
\]

\[
\omega^{n+1}=\omega^n+\alpha^n\Delta t,
\]

\[
\theta^{n+1}=\theta^n+\omega^{n+1}\Delta t.
\]

The forces and torques are evaluated from the robot state at the beginning of the step. In particular, each wall's contact force is calculated before the position and velocity are updated. The contact-history increment also uses the pre-update contact-point velocity.

If several walls are supplied, their forces and torques are summed. The same pre-integration robot state is used when evaluating each wall in a given step. Contact history is maintained separately for each wall.

The function rejects a non-positive time step. If walls are supplied, contact parameters must also be supplied.

The semi-implicit Euler method is simple and efficient, but it is not exact. Accuracy depends on the time step, especially when the contact stiffness is large. Energy conservation should therefore be assessed numerically rather than assumed.

---

## 8. Software organization

The supplied project files have the following responsibilities:

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, build configuration, dependencies |
| `src/bristlebots/robot.py` | Robot state, disk inertia, body direction, free-speed property |
| `src/bristlebots/geometry.py` | Finite wall segments and disk–wall contact geometry |
| `src/bristlebots/contacts.py` | Contact parameters/state, normal force, tangential friction, contact torque |
| `src/bristlebots/simulation.py` | Force and torque assembly, one integration step |
| `src/bristlebots/visualization.py` | Matplotlib animation |

The package metadata specifies Python 3.10 or later and depends on NumPy and Matplotlib. The build backend is setuptools, with package discovery under `src`.

Typical imports are:

```python
from bristlebots.robot import Robot
from bristlebots.geometry import Wall
from bristlebots.contacts import ContactParameters
from bristlebots.simulation import step
```

A minimal one-step setup is:

```python
robot = Robot(
    position=[0.0, 0.0],
    velocity=[1.0, 0.0],
    radius=0.5,
    mass=1.0,
    propulsion_force=0.0,
    translational_drag=0.0,
    rotational_drag=0.0,
)

wall = Wall(
    start=[2.0, -1.0],
    end=[2.0, 1.0],
)

contact_parameters = ContactParameters(
    normal_stiffness=4e5,
    normal_damping=0.0,
    mu_static=0.5,
    mu_kinetic=0.3,
    tangential_stiffness=1e4,
)

step(
    robot,
    dt=1e-6,
    walls=[wall],
    contact_parameters=contact_parameters,
)
```

This is only an illustration of the public interfaces. A full simulation loop, storage of trajectory history, and selection of physical parameters are the responsibility of the experiment script.

---

## 9. Visualization

The `animate_robots` function uses Matplotlib's `FuncAnimation` to display stored trajectory data. It accepts:

- a list of robot objects;
- `trajectory_history`, containing positions at each frame;
- optional angular-velocity and orientation histories;
- optional wall segments;
- plotting-domain size, animation interval, and trail length.

Each robot is drawn as a disk. An arrow indicates its body-axis direction, a line displays its recent trajectory, and a text label displays angular velocity. Wall segments are drawn as black line segments.

The expected trajectory-history layout is effectively

\[
(\text{number of frames},\text{number of robots},2).
\]

The orientation and angular-velocity histories are expected to have shape

\[
(\text{number of frames},\text{number of robots}).
\]

If either history is omitted, the current function substitutes zeros. Thus, omitted orientation history displays all body axes along the positive \(x\)-direction, and omitted angular-velocity history displays zero angular velocity.

**Known implementation issue:** inside the animation update callback, the code calls `ax.add_patch(arrow)` and appends `arrow` to `arrows` again. This refers to the last arrow created during initial setup, rather than creating or updating a distinct arrow for each robot. Repeated frames can therefore add duplicate references/patches. The later check `if theta_history is None` is also unreachable because `theta_history` has already been replaced by a zero array when it was initially `None`. These are visualization-code issues and do not affect the dynamics integrator.

---

## 10. Tests and what they establish

The supplied tests cover selected properties of wall contact and integration. They are useful verification checks, but they do not prove that every aspect of the model is physically complete.

### 10.1 Wall-collision tests

| Test | What it checks |
|---|---|
| `test_normal_collision` | Undamped normal collision against a linear spring is compared with the analytical harmonic-oscillator solution for penetration and horizontal velocity. |
| `test_rotation_without_friction` | With friction and drag absent, contact does not change angular velocity; orientation advances at the initial angular velocity. |
| `test_kinetic_friction` | Checks a conserved linear-angular velocity combination for the chosen frictional contact geometry while contact is active. |
| `test_time_step_convergence` | Checks that the maximum-penetration error decreases across the selected sequence of time steps. |
| `test_static_friction` | Checks that the computed tangential force stays below the static Coulomb threshold in the chosen initial state. |
| `test_kinetic_friction_limit` | Forces the kinetic branch using a large stored tangential displacement and checks the kinetic force cap and resulting torque magnitude. |
| `test_stick_slip_transition` | Checks that the implementation selects the static branch for a small initial spring displacement and the kinetic branch after a deliberately large displacement is imposed. |
| `test_normal_contact_force_with_damping` | Checks the normal spring-dashpot force against its direct analytical expression for an approaching disk. |
| `test_damped_normal_collision` | Compares numerical penetration and velocity with the underdamped spring-dashpot analytical solution. |

The stick–slip test deliberately sets a large tangential displacement. It verifies the branch-switching logic, not a naturally evolving stick-to-slip transition generated by a complete physical trajectory.

The time-step test checks monotonic improvement for the three selected time steps. It is not a general convergence-order measurement.

### 10.2 Simulation tests

| Test | What it checks |
|---|---|
| `test_robot_contact_with_corner` | Two perpendicular walls exert simultaneous normal forces; the symmetric setup produces equal positive velocity components. |
| `test_energy_conservation_during_wall_collision` | For an undamped, frictionless spring contact, kinetic plus normal spring potential energy remains within a 1% relative deviation over the selected run; the robot leaves the wall. |
| `test_tangential_spring_contact_matches_analytic_solution` | In a selected sticking regime, the coupled tangential translation and rotation are compared with an analytical spring-contact solution. |
| `test_wall_order_does_not_change_result` | Reversing the order of two walls gives the same one-step state for the symmetric frictionless setup. |

For the tangential spring test, the effective inverse mass for tangential contact is

\[
A=\frac{1}{m}+\frac{R^2}{I},
\]

and the corresponding angular frequency is

\[
\Omega_t=\sqrt{k_t A}.
\]

Under the assumptions of that test (fixed contact normal, no damping, no slip, and constant normal contact), the tangential spring couples translation and rotation. The test compares the final \(v_y\) and \(\omega\) with the analytical expressions derived for this reduced system.

The energy test is a numerical tolerance check, not exact conservation by the integrator. Its result applies to the particular parameters, initial condition, and duration used in the test.

---

## 11. Current assumptions and limitations

The current model makes the following assumptions:

1. Motion is restricted to a plane.
2. Each robot is a rigid, uniform disk.
3. The walls are finite, rigid, and immobile.
4. Active propulsion has constant magnitude and acts along the body axis through the center.
5. Translational and rotational drag are linear.
6. Contact is represented by finite-stiffness forces, allowing small numerical overlap.
7. Tangential friction uses a Cundall–Strack history spring with Coulomb limits.
8. The contact normal and tangent are calculated from the closest point on each wall segment.
9. Robot–robot contacts and many-body collision resolution are not present in the supplied implementation.

Some numerical and software limitations should also be kept in mind:

- The time step must resolve the fastest contact dynamics.
- Tangential history is not transported when the contact frame rotates.
- A contact state is keyed by the wall object; replacing wall objects during a run creates new contact histories.
- The robot constructor does not validate all physical parameters.
- `free_speed` is undefined for zero translational drag.
- The visualization callback contains the duplicate-arrow issue described above.
- No supplied test directly verifies contact-history reset after separation, long-time frictional energy dissipation, or robustness near wall endpoints under changing contact normals.

---

## 12. Suggested development sequence

The following are possible next steps, not features claimed to exist in the current code.

### Stage 1 — Strengthen single-robot contact verification

- Add a test that verifies tangential history is reset after the robot separates from a wall.
- Add system-level checks of energy change for sliding friction and normal damping.
- Test contact near finite-wall endpoints and during changes in contact normal.
- Perform systematic time-step convergence studies for both normal and tangential contact.
- Add parameter validation for positive mass and radius and non-negative drag coefficients.
- Guard the `free_speed` property when translational drag is zero.

### Stage 2 — Improve contact-history treatment

- Transport the tangential history consistently as the contact frame changes.
- Clarify and test the zero-slip transition between sticking and sliding.
- Consider whether a more advanced contact solver is needed for simultaneous contacts at corners.

### Stage 3 — Build maze experiments

- Introduce reusable wall collections and maze geometries.
- Test a straight corridor, a bend, a dead end, and more complex labyrinths.
- Record trajectories, velocity distributions, angular velocities, and density fields.
- Vary geometry independently from robot parameters.

### Stage 4 — Add robot–robot interactions

- Implement pairwise disk–disk geometry.
- Apply equal-and-opposite contact forces to satisfy Newton's third law.
- Include torques from tangential forces on both robots.
- Maintain contact history for each robot pair and test pair-order independence.
- Validate isolated two-robot collisions before increasing the number of robots.

### Stage 5 — Study collective behavior and continuum descriptions

- Simulate multiple robots in controlled geometries.
- Measure density, mean velocity, velocity fluctuations, and orientational statistics.
- Compare coarse-grained microscopic fields with candidate continuum models, including Toner–Tu.
- Determine which continuum terms are supported by the microscopic dynamics rather than inserting alignment or other interactions without physical justification.

---

## 13. Summary

The supplied code implements a two-dimensional inertial disk model with active propulsion, linear drag, compliant robot–wall contact, and Cundall–Strack tangential friction. The robot's translational and rotational motion are coupled through the torque generated by tangential contact forces. The equations are advanced using semi-implicit Euler integration, and selected analytical and symmetry-based tests check important parts of the implementation.

The current simulator is a foundation for single-robot motion and wall interactions. It does not yet implement robot–robot collisions, full maze experimentation, or continuum-model comparison. Those remain natural extensions of the research program.
