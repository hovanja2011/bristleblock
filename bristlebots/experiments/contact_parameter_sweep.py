"""Stiffness/friction parameter study for oblique bristlebots collisions.

This is a parameter study of the compliant Cundall-Strack contact model, not
an analytical solution of a frictional collision. Time-step pairs are included
at every parameter point to expose temporal-discretization sensitivity.
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bristlebots.contacts import ContactParameters
from bristlebots.robot import Robot
from bristlebots.simulation import step_robots


RADIUS = 1.0
MASS = 1.0
FINAL_TIME = 0.18
STIFFNESSES = (500.0, 1000.0, 2000.0, 4000.0)
FRICTION = (0.0, 0.1, 0.3, 0.6)
DT_FACTOR = 0.004


def _invariants(robots):
    momentum = sum((r.mass * r.velocity for r in robots), np.zeros(2))
    angular = sum(
        r.moment_of_inertia * r.omega
        + r.mass * (r.position[0]*r.velocity[1] - r.position[1]*r.velocity[0])
        for r in robots
    )
    energy = sum(
        0.5 * r.mass * np.dot(r.velocity, r.velocity)
        + 0.5 * r.moment_of_inertia * r.omega**2
        for r in robots
    )
    return momentum, float(angular), float(energy)


def run_case(normal_stiffness, friction, dt_factor=DT_FACTOR):
    """Run one collision and return dimensioned diagnostics."""
    effective_mass = MASS / 2.0
    dt = dt_factor * np.sqrt(effective_mass / normal_stiffness)
    steps = int(np.ceil(FINAL_TIME / dt))
    dt = FINAL_TIME / steps

    angle = np.deg2rad(30.0)
    normal = np.array([np.cos(angle), np.sin(angle)])
    tangent = np.array([-normal[1], normal[0]])
    relative_velocity = -0.8 * normal + 0.6 * tangent
    initial_distance = 2.05

    a = Robot(
        position=(0.0, 0.0),
        velocity=-0.5 * relative_velocity,
        omega=0.3,
        radius=RADIUS,
        mass=MASS,
    )
    b = Robot(
        position=initial_distance * normal,
        velocity=0.5 * relative_velocity,
        omega=-0.2,
        radius=RADIUS,
        mass=MASS,
    )
    robots = [a, b]

    damping_ratio = 0.05
    normal_damping = 2.0 * damping_ratio * np.sqrt(
        normal_stiffness * effective_mass
    )
    tangential_stiffness = 200.0
    parameters = ContactParameters(
        normal_stiffness=normal_stiffness,
        normal_damping=normal_damping,
        mu_static=friction,
        mu_kinetic=0.8 * friction,
        tangential_stiffness=tangential_stiffness,
        tangential_damping=0.1,
    )

    p0, l0, e0 = _invariants(robots)
    max_overlap = 0.0
    normal_impulse = 0.0
    tangential_impulse = 0.0
    contact_steps = sliding_steps = sticking_steps = 0

    for _ in range(steps):
        diagnostics = step_robots(
            robots, dt, contact_parameters=parameters
        )
        result = diagnostics[0]["robot_contact_results"][0][1]
        max_overlap = max(max_overlap, result.overlap)
        if result.overlap > 0.0 and result.normal_force_magnitude > 0.0:
            contact_steps += 1
            normal_impulse += result.normal_force_magnitude * dt
            tangential_impulse += result.tangential_force_magnitude * dt
            if result.mode == "sliding":
                sliding_steps += 1
            elif result.mode == "sticking":
                sticking_steps += 1

    pf, lf, ef = _invariants(robots)
    return {
        "normal_stiffness": normal_stiffness,
        "mu_static": friction,
        "mu_kinetic": 0.8 * friction,
        "dt": dt,
        "steps": steps,
        "max_overlap": max_overlap,
        "relative_overlap": max_overlap / RADIUS,
        "normal_impulse": normal_impulse,
        "tangential_impulse": tangential_impulse,
        "final_omega_a": a.omega,
        "final_omega_b": b.omega,
        "spin_change": abs(a.omega - 0.3) + abs(b.omega + 0.2),
        "kinetic_energy_loss": e0 - ef,
        "momentum_drift": float(np.linalg.norm(pf - p0)),
        "angular_momentum_drift": abs(lf - l0),
        "contact_steps": contact_steps,
        "sliding_steps": sliding_steps,
        "sticking_steps": sticking_steps,
    }


def run_sweep():
    rows = []
    for stiffness in STIFFNESSES:
        for friction in FRICTION:
            coarse = run_case(stiffness, friction, DT_FACTOR)
            fine = run_case(stiffness, friction, DT_FACTOR / 2.0)
            coarse["dt_refined_spin_difference"] = abs(
                coarse["final_omega_a"] - fine["final_omega_a"]
            ) + abs(coarse["final_omega_b"] - fine["final_omega_b"])
            coarse["dt_refined_overlap_difference"] = abs(
                coarse["max_overlap"] - fine["max_overlap"]
            )
            rows.append(coarse)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/validation")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows = run_sweep()
    csv_path = output / "contact_parameter_sweep.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for mu in FRICTION:
        subset = [r for r in rows if r["mu_static"] == mu]
        axes[0].loglog(
            [r["normal_stiffness"] for r in subset],
            [r["relative_overlap"] for r in subset],
            "o-", label=fr"$\mu_s={mu:g}$",
        )
        axes[1].plot(
            [r["normal_stiffness"] for r in subset],
            [r["spin_change"] for r in subset],
            "o-", label=fr"$\mu_s={mu:g}$",
        )
    axes[0].set_xlabel("Normal stiffness $k_n$")
    axes[0].set_ylabel("Maximum overlap / radius")
    axes[0].set_title("Penalty deformation")
    axes[1].set_xlabel("Normal stiffness $k_n$")
    axes[1].set_ylabel("Total absolute spin change")
    axes[1].set_title("Friction-induced rotation")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "contact_parameter_sweep.png", dpi=160)

    print("kn, mu_s, max overlap/R, spin change, dt-refined spin difference")
    for row in rows:
        print(
            f'{row["normal_stiffness"]:g}, {row["mu_static"]:g}, '
            f'{row["relative_overlap"]:.6e}, {row["spin_change"]:.6e}, '
            f'{row["dt_refined_spin_difference"]:.6e}'
        )
    print(f"Saved {csv_path} and contact_parameter_sweep.png")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
