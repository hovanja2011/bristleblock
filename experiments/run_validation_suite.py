"""Run reproducible model, contact and maze validation experiments."""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "bristlebots/experiments/free_motion_validation.py",
    "bristlebots/experiments/rotational_validation.py",
    "bristlebots/experiments/head_on_collision_validation.py",
    "bristlebots/experiments/oblique_collision_validation.py",
    "bristlebots/experiments/contact_parameter_sweep.py",
    "toner_tu/experiments/periodic_acoustic_convergence.py",
    "toner_tu/experiments/damped_acoustic_mode.py",
    "toner_tu/experiments/nonlinear_ordering_validation.py",
    "toner_tu/experiments/nonlinear_spatial_dynamics.py",
    "mazes/experiments/rounded_obstacle_validation.py",
)
TABLES = (
    "free_motion_validation.csv", "rotational_convergence.csv",
    "head_on_collision.csv", "oblique_collision.csv",
    "contact_parameter_sweep.csv", "periodic_acoustic_convergence.csv",
    "damped_acoustic_mode.csv", "nonlinear_ordering.csv",
    "nonlinear_spatial_dynamics.csv", "rounded_obstacle_validation.csv",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/validation")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for script in SCRIPTS:
        command = [sys.executable, str(ROOT/script), "--output-dir", str(output)]
        if args.no_show:
            command.append("--no-show")
        print("\n>>> " + " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    combined = []
    for filename in TABLES:
        path = output/filename
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                row["source_file"] = filename
                combined.append(row)
    keys = sorted({key for row in combined for key in row})
    with (output/"summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(combined)
    (output/"README.md").write_text(
        "# Validation suite outputs\\n\\n"
        "Experiment-specific CSV files preserve metric names and units. "
        "summary.csv concatenates rows; unlike error measures must not be "
        "compared without consulting each benchmark description.\\n\\n"
        "PNG figures show analytical/reference comparisons, parameter sweeps, "
        "geometry refinement and nonlinear field diagnostics. Interactive "
        "animations are opened by the Toner–Tu and bristlebots experiments.\\n"
    )
    print(f"Validation suite complete. Results: {output}")


if __name__ == "__main__":
    main()
